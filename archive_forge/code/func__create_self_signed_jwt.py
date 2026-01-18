import copy
import datetime
from google.auth import _helpers
from google.auth import _service_account_info
from google.auth import credentials
from google.auth import jwt
from google.oauth2 import _client
def _create_self_signed_jwt(self, audience):
    """Create a self-signed JWT from the credentials if requirements are met.

        Args:
            audience (str): The service URL. ``https://[API_ENDPOINT]/``
        """
    if self._always_use_jwt_access:
        if self._scopes:
            additional_claims = {'scope': ' '.join(self._scopes)}
            if self._jwt_credentials is None or self._jwt_credentials.additional_claims != additional_claims:
                self._jwt_credentials = jwt.Credentials.from_signing_credentials(self, None, additional_claims=additional_claims)
        elif audience:
            if self._jwt_credentials is None or self._jwt_credentials._audience != audience:
                self._jwt_credentials = jwt.Credentials.from_signing_credentials(self, audience)
        elif self._default_scopes:
            additional_claims = {'scope': ' '.join(self._default_scopes)}
            if self._jwt_credentials is None or additional_claims != self._jwt_credentials.additional_claims:
                self._jwt_credentials = jwt.Credentials.from_signing_credentials(self, None, additional_claims=additional_claims)
    elif not self._scopes and audience:
        self._jwt_credentials = jwt.Credentials.from_signing_credentials(self, audience)