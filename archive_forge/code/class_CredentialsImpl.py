import datetime
import pytest  # type: ignore
from google.auth import _credentials_async as credentials
from google.auth import _helpers
class CredentialsImpl(credentials.Credentials):

    def refresh(self, request):
        self.token = request

    def with_quota_project(self, quota_project_id):
        raise NotImplementedError()