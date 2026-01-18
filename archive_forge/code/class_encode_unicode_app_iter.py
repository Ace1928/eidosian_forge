import io
import sys
import warnings
from traceback import print_exception
from io import StringIO
from urllib.parse import unquote, urlsplit
from paste.request import get_cookies, parse_querystring, parse_formvars
from paste.request import construct_url, path_info_split, path_info_pop
from paste.response import HeaderDict, has_header, header_value, remove_header
from paste.response import error_body_response, error_response, error_response_app
class encode_unicode_app_iter(object):
    """
    Encodes an app_iterable's unicode responses as strings
    """

    def __init__(self, app_iterable, encoding=sys.getdefaultencoding(), errors='strict'):
        self.app_iterable = app_iterable
        self.app_iter = iter(app_iterable)
        self.encoding = encoding
        self.errors = errors

    def __iter__(self):
        return self

    def next(self):
        content = next(self.app_iter)
        if isinstance(content, str):
            content = content.encode(self.encoding, self.errors)
        return content
    __next__ = next

    def close(self):
        if hasattr(self.app_iterable, 'close'):
            self.app_iterable.close()