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
def CredentialsFromP12Key(private_key, account, password=None):
    """Creates credentials object from given p12 private key and account name."""
    return p12_service_account.CreateP12ServiceAccount(private_key, password, service_account_email=account, token_uri=c_creds.GetEffectiveTokenUri({}), scopes=config.CLOUDSDK_SCOPES)