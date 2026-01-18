import logging
import string
from collections import defaultdict
from urllib.parse import urlparse
from django.conf import settings
from django.core.exceptions import DisallowedHost, ImproperlyConfigured
from django.http import HttpHeaders, UnreadablePostError
from django.urls import get_callable
from django.utils.cache import patch_vary_headers
from django.utils.crypto import constant_time_compare, get_random_string
from django.utils.deprecation import MiddlewareMixin
from django.utils.functional import cached_property
from django.utils.http import is_same_domain
from django.utils.log import log_response
from django.utils.regex_helper import _lazy_re_compile
class CsrfViewMiddleware(MiddlewareMixin):
    """
    Require a present and correct csrfmiddlewaretoken for POST requests that
    have a CSRF cookie, and set an outgoing CSRF cookie.

    This middleware should be used in conjunction with the {% csrf_token %}
    template tag.
    """

    @cached_property
    def csrf_trusted_origins_hosts(self):
        return [urlparse(origin).netloc.lstrip('*') for origin in settings.CSRF_TRUSTED_ORIGINS]

    @cached_property
    def allowed_origins_exact(self):
        return {origin for origin in settings.CSRF_TRUSTED_ORIGINS if '*' not in origin}

    @cached_property
    def allowed_origin_subdomains(self):
        """
        A mapping of allowed schemes to list of allowed netlocs, where all
        subdomains of the netloc are allowed.
        """
        allowed_origin_subdomains = defaultdict(list)
        for parsed in (urlparse(origin) for origin in settings.CSRF_TRUSTED_ORIGINS if '*' in origin):
            allowed_origin_subdomains[parsed.scheme].append(parsed.netloc.lstrip('*'))
        return allowed_origin_subdomains

    def _accept(self, request):
        request.csrf_processing_done = True
        return None

    def _reject(self, request, reason):
        response = _get_failure_view()(request, reason=reason)
        log_response('Forbidden (%s): %s', reason, request.path, response=response, request=request, logger=logger)
        return response

    def _get_secret(self, request):
        """
        Return the CSRF secret originally associated with the request, or None
        if it didn't have one.

        If the CSRF_USE_SESSIONS setting is false, raises InvalidTokenFormat if
        the request's secret has invalid characters or an invalid length.
        """
        if settings.CSRF_USE_SESSIONS:
            try:
                csrf_secret = request.session.get(CSRF_SESSION_KEY)
            except AttributeError:
                raise ImproperlyConfigured('CSRF_USE_SESSIONS is enabled, but request.session is not set. SessionMiddleware must appear before CsrfViewMiddleware in MIDDLEWARE.')
        else:
            try:
                csrf_secret = request.COOKIES[settings.CSRF_COOKIE_NAME]
            except KeyError:
                csrf_secret = None
            else:
                _check_token_format(csrf_secret)
        if csrf_secret is None:
            return None
        if len(csrf_secret) == CSRF_TOKEN_LENGTH:
            csrf_secret = _unmask_cipher_token(csrf_secret)
        return csrf_secret

    def _set_csrf_cookie(self, request, response):
        if settings.CSRF_USE_SESSIONS:
            if request.session.get(CSRF_SESSION_KEY) != request.META['CSRF_COOKIE']:
                request.session[CSRF_SESSION_KEY] = request.META['CSRF_COOKIE']
        else:
            response.set_cookie(settings.CSRF_COOKIE_NAME, request.META['CSRF_COOKIE'], max_age=settings.CSRF_COOKIE_AGE, domain=settings.CSRF_COOKIE_DOMAIN, path=settings.CSRF_COOKIE_PATH, secure=settings.CSRF_COOKIE_SECURE, httponly=settings.CSRF_COOKIE_HTTPONLY, samesite=settings.CSRF_COOKIE_SAMESITE)
            patch_vary_headers(response, ('Cookie',))

    def _origin_verified(self, request):
        request_origin = request.META['HTTP_ORIGIN']
        try:
            good_host = request.get_host()
        except DisallowedHost:
            pass
        else:
            good_origin = '%s://%s' % ('https' if request.is_secure() else 'http', good_host)
            if request_origin == good_origin:
                return True
        if request_origin in self.allowed_origins_exact:
            return True
        try:
            parsed_origin = urlparse(request_origin)
        except ValueError:
            return False
        request_scheme = parsed_origin.scheme
        request_netloc = parsed_origin.netloc
        return any((is_same_domain(request_netloc, host) for host in self.allowed_origin_subdomains.get(request_scheme, ())))

    def _check_referer(self, request):
        referer = request.META.get('HTTP_REFERER')
        if referer is None:
            raise RejectRequest(REASON_NO_REFERER)
        try:
            referer = urlparse(referer)
        except ValueError:
            raise RejectRequest(REASON_MALFORMED_REFERER)
        if '' in (referer.scheme, referer.netloc):
            raise RejectRequest(REASON_MALFORMED_REFERER)
        if referer.scheme != 'https':
            raise RejectRequest(REASON_INSECURE_REFERER)
        if any((is_same_domain(referer.netloc, host) for host in self.csrf_trusted_origins_hosts)):
            return
        good_referer = settings.SESSION_COOKIE_DOMAIN if settings.CSRF_USE_SESSIONS else settings.CSRF_COOKIE_DOMAIN
        if good_referer is None:
            try:
                good_referer = request.get_host()
            except DisallowedHost:
                raise RejectRequest(REASON_BAD_REFERER % referer.geturl())
        else:
            server_port = request.get_port()
            if server_port not in ('443', '80'):
                good_referer = '%s:%s' % (good_referer, server_port)
        if not is_same_domain(referer.netloc, good_referer):
            raise RejectRequest(REASON_BAD_REFERER % referer.geturl())

    def _bad_token_message(self, reason, token_source):
        if token_source != 'POST':
            header_name = HttpHeaders.parse_header_name(token_source)
            token_source = f'the {header_name!r} HTTP header'
        return f'CSRF token from {token_source} {reason}.'

    def _check_token(self, request):
        try:
            csrf_secret = self._get_secret(request)
        except InvalidTokenFormat as exc:
            raise RejectRequest(f'CSRF cookie {exc.reason}.')
        if csrf_secret is None:
            raise RejectRequest(REASON_NO_CSRF_COOKIE)
        request_csrf_token = ''
        if request.method == 'POST':
            try:
                request_csrf_token = request.POST.get('csrfmiddlewaretoken', '')
            except UnreadablePostError:
                pass
        if request_csrf_token == '':
            try:
                request_csrf_token = request.META[settings.CSRF_HEADER_NAME]
            except KeyError:
                raise RejectRequest(REASON_CSRF_TOKEN_MISSING)
            token_source = settings.CSRF_HEADER_NAME
        else:
            token_source = 'POST'
        try:
            _check_token_format(request_csrf_token)
        except InvalidTokenFormat as exc:
            reason = self._bad_token_message(exc.reason, token_source)
            raise RejectRequest(reason)
        if not _does_token_match(request_csrf_token, csrf_secret):
            reason = self._bad_token_message('incorrect', token_source)
            raise RejectRequest(reason)

    def process_request(self, request):
        try:
            csrf_secret = self._get_secret(request)
        except InvalidTokenFormat:
            _add_new_csrf_cookie(request)
        else:
            if csrf_secret is not None:
                request.META['CSRF_COOKIE'] = csrf_secret

    def process_view(self, request, callback, callback_args, callback_kwargs):
        if getattr(request, 'csrf_processing_done', False):
            return None
        if getattr(callback, 'csrf_exempt', False):
            return None
        if request.method in ('GET', 'HEAD', 'OPTIONS', 'TRACE'):
            return self._accept(request)
        if getattr(request, '_dont_enforce_csrf_checks', False):
            return self._accept(request)
        if 'HTTP_ORIGIN' in request.META:
            if not self._origin_verified(request):
                return self._reject(request, REASON_BAD_ORIGIN % request.META['HTTP_ORIGIN'])
        elif request.is_secure():
            try:
                self._check_referer(request)
            except RejectRequest as exc:
                return self._reject(request, exc.reason)
        try:
            self._check_token(request)
        except RejectRequest as exc:
            return self._reject(request, exc.reason)
        return self._accept(request)

    def process_response(self, request, response):
        if request.META.get('CSRF_COOKIE_NEEDS_UPDATE'):
            self._set_csrf_cookie(request, response)
            request.META['CSRF_COOKIE_NEEDS_UPDATE'] = False
        return response