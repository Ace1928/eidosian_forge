import aiohttp  # type: ignore
from aioresponses import aioresponses, core  # type: ignore
import mock
import pytest  # type: ignore
from tests_async.transport import async_compliance
import google.auth._credentials_async
from google.auth.transport import _aiohttp_requests as aiohttp_requests
import google.auth.transport._mtls_helper
class CredentialsStub(google.auth._credentials_async.Credentials):

    def __init__(self, token='token'):
        super(CredentialsStub, self).__init__()
        self.token = token

    def apply(self, headers, token=None):
        headers['authorization'] = self.token

    def refresh(self, request):
        self.token += '1'