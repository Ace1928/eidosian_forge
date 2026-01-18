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
def _RaiseCredentialsCorrupt(self, e: 'BaseException') -> None:
    bq_utils.ProcessError(e, name='GetCredentialsFromFlags', message_prefix='Credentials appear corrupt. Please delete the credential file and try your command again. You can delete your credential file using "bq init --delete_credentials".\n\nIf that does not work, you may have encountered a bug in the BigQuery CLI.')
    sys.exit(1)