import argparse
import json
import logging
import os
import sys
from typing import List, Optional, Union
from absl import app
from absl import flags
from google_reauth.reauth_creds import Oauth2WithReauthCredentials
import httplib2
import oauth2client_4_0
import oauth2client_4_0.contrib
import oauth2client_4_0.contrib.gce
import oauth2client_4_0.contrib.multiprocess_file_storage
import oauth2client_4_0.file
import oauth2client_4_0.service_account
import oauth2client_4_0.tools
import requests
import bq_auth_flags
import bq_utils
import wrapped_credentials
from utils import bq_error
class ApplicationDefaultCredentialFileLoader(CachedCredentialLoader):
    """Credential loader for application default credential file."""

    def __init__(self, credential_file: str, *args, **kwargs) -> None:
        """Creates ApplicationDefaultCredentialFileLoader instance.

    Args:
      credential_file: path to credential file in json format.
      *args: additional arguments to apply to base class.
      **kwargs: additional keyword arguments to apply to base class.
    """
        super(ApplicationDefaultCredentialFileLoader, self).__init__(*args, **kwargs)
        self._credential_file = credential_file

    def _Load(self) -> WrappedCredentialsUnionType:
        """Loads credentials from given application default credential file."""
        with open(self._credential_file) as file_obj:
            credentials = json.load(file_obj)
        client_scope = bq_utils.GetClientScopesFromFlags()
        if credentials['type'] == oauth2client_4_0.client.AUTHORIZED_USER:
            return Oauth2WithReauthCredentials(access_token=None, client_id=credentials['client_id'], client_secret=credentials['client_secret'], refresh_token=credentials['refresh_token'], token_expiry=None, token_uri=oauth2client_4_0.GOOGLE_TOKEN_URI, user_agent=_CLIENT_USER_AGENT, scopes=client_scope)
        elif credentials['type'] == 'external_account':
            return wrapped_credentials.WrappedCredentials.for_external_account(self._credential_file)
        elif credentials['type'] == 'external_account_authorized_user':
            return wrapped_credentials.WrappedCredentials.for_external_account_authorized_user(self._credential_file)
        else:
            credentials['type'] = oauth2client_4_0.client.SERVICE_ACCOUNT
            service_account_credentials = oauth2client_4_0.service_account.ServiceAccountCredentials.from_json_keyfile_dict(keyfile_dict=credentials, scopes=client_scope)
            service_account_credentials._user_agent = _CLIENT_USER_AGENT
            return service_account_credentials