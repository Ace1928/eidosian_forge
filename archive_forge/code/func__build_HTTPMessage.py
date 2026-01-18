import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def _build_HTTPMessage(self, raw_headers):
    status_and_headers = BytesIO(raw_headers)
    status_and_headers.readline()
    msg = parse_headers(status_and_headers)
    return msg.get