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
def _GetCredentialsLoaderFromFlags() -> 'CachedCredentialLoader | AccessTokenCredentialLoader':
    """Returns a CredentialsLoader based on user-supplied flags."""
    if FLAGS.oauth_access_token:
        logging.info('Loading credentials using oauth_access_token')
        return AccessTokenCredentialLoader(access_token=FLAGS.oauth_access_token)
    if FLAGS.service_account:
        logging.info('Loading credentials using service_account')
        if not FLAGS.service_account_credential_file:
            raise app.UsageError('The flag --service_account_credential_file must be specified if --service_account is used.')
        if FLAGS.service_account_private_key_file:
            logging.info('Loading credentials using service_account_private_key_file')
            return ServiceAccountPrivateKeyFileLoader(credential_cache_file=FLAGS.service_account_credential_file, read_cache_first=True, service_account=FLAGS.service_account, file_path=FLAGS.service_account_private_key_file, password=FLAGS.service_account_private_key_password)
        raise app.UsageError('Service account authorization requires --service_account_private_key_file flag to be set.')
    if FLAGS.application_default_credential_file:
        logging.info('Loading credentials using application_default_credential_file')
        if not FLAGS.credential_file:
            raise app.UsageError('The flag --credential_file must be specified if --application_default_credential_file is used.')
        return ApplicationDefaultCredentialFileLoader(credential_cache_file=FLAGS.credential_file, read_cache_first=True, credential_file=FLAGS.application_default_credential_file)
    raise app.UsageError('bq.py should not be invoked. Use bq command instead.')