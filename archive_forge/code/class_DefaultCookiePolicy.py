import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
class DefaultCookiePolicy(CookiePolicy):
    """Implements the standard rules for accepting and returning cookies."""
    DomainStrictNoDots = 1
    DomainStrictNonDomain = 2
    DomainRFC2965Match = 4
    DomainLiberal = 0
    DomainStrict = DomainStrictNoDots | DomainStrictNonDomain

    def __init__(self, blocked_domains=None, allowed_domains=None, netscape=True, rfc2965=False, rfc2109_as_netscape=None, hide_cookie2=False, strict_domain=False, strict_rfc2965_unverifiable=True, strict_ns_unverifiable=False, strict_ns_domain=DomainLiberal, strict_ns_set_initial_dollar=False, strict_ns_set_path=False, secure_protocols=('https', 'wss')):
        """Constructor arguments should be passed as keyword arguments only."""
        self.netscape = netscape
        self.rfc2965 = rfc2965
        self.rfc2109_as_netscape = rfc2109_as_netscape
        self.hide_cookie2 = hide_cookie2
        self.strict_domain = strict_domain
        self.strict_rfc2965_unverifiable = strict_rfc2965_unverifiable
        self.strict_ns_unverifiable = strict_ns_unverifiable
        self.strict_ns_domain = strict_ns_domain
        self.strict_ns_set_initial_dollar = strict_ns_set_initial_dollar
        self.strict_ns_set_path = strict_ns_set_path
        self.secure_protocols = secure_protocols
        if blocked_domains is not None:
            self._blocked_domains = tuple(blocked_domains)
        else:
            self._blocked_domains = ()
        if allowed_domains is not None:
            allowed_domains = tuple(allowed_domains)
        self._allowed_domains = allowed_domains

    def blocked_domains(self):
        """Return the sequence of blocked domains (as a tuple)."""
        return self._blocked_domains

    def set_blocked_domains(self, blocked_domains):
        """Set the sequence of blocked domains."""
        self._blocked_domains = tuple(blocked_domains)

    def is_blocked(self, domain):
        for blocked_domain in self._blocked_domains:
            if user_domain_match(domain, blocked_domain):
                return True
        return False

    def allowed_domains(self):
        """Return None, or the sequence of allowed domains (as a tuple)."""
        return self._allowed_domains

    def set_allowed_domains(self, allowed_domains):
        """Set the sequence of allowed domains, or None."""
        if allowed_domains is not None:
            allowed_domains = tuple(allowed_domains)
        self._allowed_domains = allowed_domains

    def is_not_allowed(self, domain):
        if self._allowed_domains is None:
            return False
        for allowed_domain in self._allowed_domains:
            if user_domain_match(domain, allowed_domain):
                return False
        return True

    def set_ok(self, cookie, request):
        """
        If you override .set_ok(), be sure to call this method.  If it returns
        false, so should your subclass (assuming your subclass wants to be more
        strict about which cookies to accept).

        """
        _debug(' - checking cookie %s=%s', cookie.name, cookie.value)
        assert cookie.name is not None
        for n in ('version', 'verifiability', 'name', 'path', 'domain', 'port'):
            fn_name = 'set_ok_' + n
            fn = getattr(self, fn_name)
            if not fn(cookie, request):
                return False
        return True

    def set_ok_version(self, cookie, request):
        if cookie.version is None:
            _debug('   Set-Cookie2 without version attribute (%s=%s)', cookie.name, cookie.value)
            return False
        if cookie.version > 0 and (not self.rfc2965):
            _debug('   RFC 2965 cookies are switched off')
            return False
        elif cookie.version == 0 and (not self.netscape):
            _debug('   Netscape cookies are switched off')
            return False
        return True

    def set_ok_verifiability(self, cookie, request):
        if request.unverifiable and is_third_party(request):
            if cookie.version > 0 and self.strict_rfc2965_unverifiable:
                _debug('   third-party RFC 2965 cookie during unverifiable transaction')
                return False
            elif cookie.version == 0 and self.strict_ns_unverifiable:
                _debug('   third-party Netscape cookie during unverifiable transaction')
                return False
        return True

    def set_ok_name(self, cookie, request):
        if cookie.version == 0 and self.strict_ns_set_initial_dollar and cookie.name.startswith('$'):
            _debug("   illegal name (starts with '$'): '%s'", cookie.name)
            return False
        return True

    def set_ok_path(self, cookie, request):
        if cookie.path_specified:
            req_path = request_path(request)
            if (cookie.version > 0 or (cookie.version == 0 and self.strict_ns_set_path)) and (not self.path_return_ok(cookie.path, request)):
                _debug('   path attribute %s is not a prefix of request path %s', cookie.path, req_path)
                return False
        return True

    def set_ok_domain(self, cookie, request):
        if self.is_blocked(cookie.domain):
            _debug('   domain %s is in user block-list', cookie.domain)
            return False
        if self.is_not_allowed(cookie.domain):
            _debug('   domain %s is not in user allow-list', cookie.domain)
            return False
        if cookie.domain_specified:
            req_host, erhn = eff_request_host(request)
            domain = cookie.domain
            if self.strict_domain and domain.count('.') >= 2:
                i = domain.rfind('.')
                j = domain.rfind('.', 0, i)
                if j == 0:
                    tld = domain[i + 1:]
                    sld = domain[j + 1:i]
                    if sld.lower() in ('co', 'ac', 'com', 'edu', 'org', 'net', 'gov', 'mil', 'int', 'aero', 'biz', 'cat', 'coop', 'info', 'jobs', 'mobi', 'museum', 'name', 'pro', 'travel', 'eu') and len(tld) == 2:
                        _debug('   country-code second level domain %s', domain)
                        return False
            if domain.startswith('.'):
                undotted_domain = domain[1:]
            else:
                undotted_domain = domain
            embedded_dots = undotted_domain.find('.') >= 0
            if not embedded_dots and (not erhn.endswith('.local')):
                _debug('   non-local domain %s contains no embedded dot', domain)
                return False
            if cookie.version == 0:
                if not (erhn.endswith(domain) or erhn.endswith(f'{undotted_domain}.local')) and (not erhn.startswith('.') and (not ('.' + erhn).endswith(domain))):
                    _debug('   effective request-host %s (even with added initial dot) does not end with %s', erhn, domain)
                    return False
            if cookie.version > 0 or self.strict_ns_domain & self.DomainRFC2965Match:
                if not domain_match(erhn, domain):
                    _debug('   effective request-host %s does not domain-match %s', erhn, domain)
                    return False
            if cookie.version > 0 or self.strict_ns_domain & self.DomainStrictNoDots:
                host_prefix = req_host[:-len(domain)]
                if host_prefix.find('.') >= 0 and (not IPV4_RE.search(req_host)):
                    _debug('   host prefix %s for domain %s contains a dot', host_prefix, domain)
                    return False
        return True

    def set_ok_port(self, cookie, request):
        if cookie.port_specified:
            req_port = request_port(request)
            if req_port is None:
                req_port = '80'
            else:
                req_port = str(req_port)
            for p in cookie.port.split(','):
                try:
                    int(p)
                except ValueError:
                    _debug('   bad port %s (not numeric)', p)
                    return False
                if p == req_port:
                    break
            else:
                _debug('   request port (%s) not found in %s', req_port, cookie.port)
                return False
        return True

    def return_ok(self, cookie, request):
        """
        If you override .return_ok(), be sure to call this method.  If it
        returns false, so should your subclass (assuming your subclass wants to
        be more strict about which cookies to return).

        """
        _debug(' - checking cookie %s=%s', cookie.name, cookie.value)
        for n in ('version', 'verifiability', 'secure', 'expires', 'port', 'domain'):
            fn_name = 'return_ok_' + n
            fn = getattr(self, fn_name)
            if not fn(cookie, request):
                return False
        return True

    def return_ok_version(self, cookie, request):
        if cookie.version > 0 and (not self.rfc2965):
            _debug('   RFC 2965 cookies are switched off')
            return False
        elif cookie.version == 0 and (not self.netscape):
            _debug('   Netscape cookies are switched off')
            return False
        return True

    def return_ok_verifiability(self, cookie, request):
        if request.unverifiable and is_third_party(request):
            if cookie.version > 0 and self.strict_rfc2965_unverifiable:
                _debug('   third-party RFC 2965 cookie during unverifiable transaction')
                return False
            elif cookie.version == 0 and self.strict_ns_unverifiable:
                _debug('   third-party Netscape cookie during unverifiable transaction')
                return False
        return True

    def return_ok_secure(self, cookie, request):
        if cookie.secure and request.type not in self.secure_protocols:
            _debug('   secure cookie with non-secure request')
            return False
        return True

    def return_ok_expires(self, cookie, request):
        if cookie.is_expired(self._now):
            _debug('   cookie expired')
            return False
        return True

    def return_ok_port(self, cookie, request):
        if cookie.port:
            req_port = request_port(request)
            if req_port is None:
                req_port = '80'
            for p in cookie.port.split(','):
                if p == req_port:
                    break
            else:
                _debug('   request port %s does not match cookie port %s', req_port, cookie.port)
                return False
        return True

    def return_ok_domain(self, cookie, request):
        req_host, erhn = eff_request_host(request)
        domain = cookie.domain
        if domain and (not domain.startswith('.')):
            dotdomain = '.' + domain
        else:
            dotdomain = domain
        if cookie.version == 0 and self.strict_ns_domain & self.DomainStrictNonDomain and (not cookie.domain_specified) and (domain != erhn):
            _debug('   cookie with unspecified domain does not string-compare equal to request domain')
            return False
        if cookie.version > 0 and (not domain_match(erhn, domain)):
            _debug('   effective request-host name %s does not domain-match RFC 2965 cookie domain %s', erhn, domain)
            return False
        if cookie.version == 0 and (not ('.' + erhn).endswith(dotdomain)):
            _debug('   request-host %s does not match Netscape cookie domain %s', req_host, domain)
            return False
        return True

    def domain_return_ok(self, domain, request):
        req_host, erhn = eff_request_host(request)
        if not req_host.startswith('.'):
            req_host = '.' + req_host
        if not erhn.startswith('.'):
            erhn = '.' + erhn
        if domain and (not domain.startswith('.')):
            dotdomain = '.' + domain
        else:
            dotdomain = domain
        if not (req_host.endswith(dotdomain) or erhn.endswith(dotdomain)):
            return False
        if self.is_blocked(domain):
            _debug('   domain %s is in user block-list', domain)
            return False
        if self.is_not_allowed(domain):
            _debug('   domain %s is not in user allow-list', domain)
            return False
        return True

    def path_return_ok(self, path, request):
        _debug('- checking cookie path=%s', path)
        req_path = request_path(request)
        pathlen = len(path)
        if req_path == path:
            return True
        elif req_path.startswith(path) and (path.endswith('/') or req_path[pathlen:pathlen + 1] == '/'):
            return True
        _debug('  %s does not path-match %s', req_path, path)
        return False