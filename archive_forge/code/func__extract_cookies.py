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
def _extract_cookies(request, response, cookies):
    """Add cookies to the response.

    Cookies in requests are extracted from the headers in the original_response
    httplib.HTTPMessage which we don't create so we have to do this step
    manually.
    """
    response.cookies.extract_cookies(MockResponse(response.raw.headers), MockRequest(request))
    if cookies:
        merge_cookies(response.cookies, cookies)