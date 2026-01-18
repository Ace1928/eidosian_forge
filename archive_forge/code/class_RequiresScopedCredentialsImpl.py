import datetime
import pytest  # type: ignore
from google.auth import _credentials_async as credentials
from google.auth import _helpers
class RequiresScopedCredentialsImpl(credentials.Scoped, CredentialsImpl):

    def __init__(self, scopes=None):
        super(RequiresScopedCredentialsImpl, self).__init__()
        self._scopes = scopes

    @property
    def requires_scopes(self):
        return not self.scopes

    def with_scopes(self, scopes):
        return RequiresScopedCredentialsImpl(scopes=scopes)