from suds.properties import Unskin
from suds.transport import *
import base64
from http.cookiejar import CookieJar
import http.client
import socket
import sys
import urllib.request, urllib.error, urllib.parse
import gzip
import zlib
from logging import getLogger
@staticmethod
def __get_request_url_for_urllib(request):
    """
        Returns the given request's URL, properly encoded for use with urllib.

        We expect that the given request object already verified that the URL
        contains ASCII characters only and stored it as a native str value.

        urllib accepts URL information as a native str value and may break
        unexpectedly if given URL information in another format.

        Python 3.x httplib.client implementation must be given a unicode string
        and not a bytes object and the given string is internally converted to
        a bytes object using an explicitly specified ASCII encoding.

        Python 2.7 httplib implementation expects the URL passed to it to not
        be a unicode string. If it is, then passing it to the underlying
        httplib Request object will cause that object to forcefully convert all
        of its data to unicode, assuming that data contains ASCII data only and
        raising a UnicodeDecodeError exception if it does not (caused by simple
        unicode + string concatenation).

        Python 2.4 httplib implementation does not really care about this as it
        does not use the internal optimization present in the Python 2.7
        implementation causing all the requested data to be converted to
        unicode.

        """
    assert isinstance(request.url, str)
    return request.url