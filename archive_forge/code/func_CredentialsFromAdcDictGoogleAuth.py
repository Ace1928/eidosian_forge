from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import p12_service_account
from googlecloudsdk.core.util import files
from oauth2client import service_account
def CredentialsFromAdcDictGoogleAuth(json_key):
    """Creates google-auth creds from a dict of application default creds."""
    from google.oauth2 import service_account as google_auth_service_account
    if 'client_email' not in json_key:
        raise BadCredentialJsonFileException('The .json key file is not in a valid format.')
    json_key['token_uri'] = c_creds.GetEffectiveTokenUri(json_key)
    service_account_credentials = google_auth_service_account.Credentials.from_service_account_info
    creds = service_account_credentials(json_key, scopes=config.CLOUDSDK_SCOPES)
    creds.private_key = json_key.get('private_key')
    creds.private_key_id = json_key.get('private_key_id')
    creds.client_id = json_key.get('client_id')
    return creds