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
class ResponseHeaderDict(HeaderDict):

    def __init__(self, *args, **kw):
        warnings.warn('The class wsgilib.ResponseHeaderDict has been moved to paste.response.HeaderDict', DeprecationWarning, 2)
        HeaderDict.__init__(self, *args, **kw)