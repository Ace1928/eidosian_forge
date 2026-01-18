import copy
import datetime
from google.auth import _helpers
from google.auth import _service_account_info
from google.auth import credentials
from google.auth import jwt
from google.oauth2 import _client
def _refresh_with_iam_endpoint(self, request):
    """Use IAM generateIdToken endpoint to obtain an ID token.

        It works as follows:

        1. First we create a self signed jwt with
        https://www.googleapis.com/auth/iam being the scope.

        2. Next we use the self signed jwt as the access token, and make a POST
        request to IAM generateIdToken endpoint. The request body is:
            {
                "audience": self._target_audience,
                "includeEmail": "true",
                "useEmailAzp": "true",
            }

        If the request is succesfully, it will return {"token":"the ID token"},
        and we can extract the ID token and compute its expiry.
        """
    jwt_credentials = jwt.Credentials.from_signing_credentials(self, None, additional_claims={'scope': 'https://www.googleapis.com/auth/iam'})
    jwt_credentials.refresh(request)
    self.token, self.expiry = _client.call_iam_generate_id_token_endpoint(request, self.signer_email, self._target_audience, jwt_credentials.token.decode())