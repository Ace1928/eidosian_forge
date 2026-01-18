from io import IOBase
from django.conf import settings
from django.core import signals
from django.core.handlers import base
from django.http import HttpRequest, QueryDict, parse_cookie
from django.urls import set_script_prefix
from django.utils.encoding import repercent_broken_unicode
from django.utils.functional import cached_property
from django.utils.regex_helper import _lazy_re_compile
class WSGIRequest(HttpRequest):

    def __init__(self, environ):
        script_name = get_script_name(environ)
        path_info = get_path_info(environ) or '/'
        self.environ = environ
        self.path_info = path_info
        self.path = '%s/%s' % (script_name.rstrip('/'), path_info.replace('/', '', 1))
        self.META = environ
        self.META['PATH_INFO'] = path_info
        self.META['SCRIPT_NAME'] = script_name
        self.method = environ['REQUEST_METHOD'].upper()
        self._set_content_type_params(environ)
        try:
            content_length = int(environ.get('CONTENT_LENGTH'))
        except (ValueError, TypeError):
            content_length = 0
        self._stream = LimitedStream(self.environ['wsgi.input'], content_length)
        self._read_started = False
        self.resolver_match = None

    def _get_scheme(self):
        return self.environ.get('wsgi.url_scheme')

    @cached_property
    def GET(self):
        raw_query_string = get_bytes_from_wsgi(self.environ, 'QUERY_STRING', '')
        return QueryDict(raw_query_string, encoding=self._encoding)

    def _get_post(self):
        if not hasattr(self, '_post'):
            self._load_post_and_files()
        return self._post

    def _set_post(self, post):
        self._post = post

    @cached_property
    def COOKIES(self):
        raw_cookie = get_str_from_wsgi(self.environ, 'HTTP_COOKIE', '')
        return parse_cookie(raw_cookie)

    @property
    def FILES(self):
        if not hasattr(self, '_files'):
            self._load_post_and_files()
        return self._files
    POST = property(_get_post, _set_post)