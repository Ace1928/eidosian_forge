import io
import http.client
import json as jsonutils
from requests.adapters import HTTPAdapter
from requests.cookies import MockRequest, MockResponse
from requests.cookies import RequestsCookieJar
from requests.cookies import merge_cookies, cookiejar_from_dict
from requests.utils import get_encoding_from_headers
from urllib3.response import HTTPResponse
from requests_mock import exceptions
def _check_body_arguments(**kwargs):
    provided = [x for x in _BODY_ARGS if kwargs.pop(x, None) is not None]
    if len(provided) > 1:
        raise RuntimeError('You may only supply one body element. You supplied %s' % ', '.join(provided))
    extra = [x for x in kwargs if x not in _HTTP_ARGS]
    if extra:
        raise TypeError('Too many arguments provided. Unexpected arguments %s.' % ', '.join(extra))