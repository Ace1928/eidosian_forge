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
def CredentialsFromAdcDict(json_key):
    """Creates oauth2client creds from a dict of application default creds."""
    if 'client_email' not in json_key:
        raise BadCredentialJsonFileException('The .json key file is not in a valid format.')
    json_key['token_uri'] = c_creds.GetEffectiveTokenUri(json_key)
    creds = service_account.ServiceAccountCredentials.from_json_keyfile_dict(json_key, scopes=config.CLOUDSDK_SCOPES)
    creds.user_agent = creds._user_agent = config.CLOUDSDK_USER_AGENT
    return creds