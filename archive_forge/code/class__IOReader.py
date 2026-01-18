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
class _IOReader(io.BytesIO):
    """A reader that makes a BytesIO look like a HTTPResponse.

    A HTTPResponse will return an empty string when you read from it after
    the socket has been closed. A BytesIO will raise a ValueError. For
    compatibility we want to do the same thing a HTTPResponse does.
    """

    def read(self, *args, **kwargs):
        if self.closed:
            return b''
        if len(args) > 0 and args[0] == 0:
            return b''
        result = io.BytesIO.read(self, *args, **kwargs)
        if result == b'':
            self.close()
        return result