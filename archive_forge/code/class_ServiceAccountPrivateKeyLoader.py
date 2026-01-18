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
class ServiceAccountPrivateKeyLoader(CachedCredentialLoader):
    """Base class for loading credential from service account."""

    def Load(self) -> WrappedCredentialsUnionType:
        if not oauth2client_4_0.client.HAS_OPENSSL:
            raise app.UsageError('BigQuery requires OpenSSL to be installed in order to use service account credentials. Please install OpenSSL and the Python OpenSSL package.')
        return super(ServiceAccountPrivateKeyLoader, self).Load()