from google.auth import _credentials_async as credentials
from google.auth import _helpers
from google.auth import exceptions
from google.oauth2 import _reauth_async as reauth
from google.oauth2 import credentials as oauth2_credentials
class UserAccessTokenCredentials(oauth2_credentials.UserAccessTokenCredentials):
    """Access token credentials for user account.

    Obtain the access token for a given user account or the current active
    user account with the ``gcloud auth print-access-token`` command.

    Args:
        account (Optional[str]): Account to get the access token for. If not
            specified, the current active account will be used.
        quota_project_id (Optional[str]): The project ID used for quota
            and billing.

    """