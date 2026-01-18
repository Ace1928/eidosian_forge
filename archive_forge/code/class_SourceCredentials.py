import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import credentials
from google.auth import downscoped
from google.auth import exceptions
from google.auth import transport
class SourceCredentials(credentials.Credentials):

    def __init__(self, raise_error=False, expires_in=3600):
        super(SourceCredentials, self).__init__()
        self._counter = 0
        self._raise_error = raise_error
        self._expires_in = expires_in

    def refresh(self, request):
        if self._raise_error:
            raise exceptions.RefreshError('Failed to refresh access token in source credentials.')
        now = _helpers.utcnow()
        self._counter += 1
        self.token = 'ACCESS_TOKEN_{}'.format(self._counter)
        self.expiry = now + datetime.timedelta(seconds=self._expires_in)