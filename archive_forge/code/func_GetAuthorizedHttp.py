from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import enum
from http import client as http_client_lib
import json
import logging
import re
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Optional, Union
import urllib
from absl import flags
import googleapiclient
from googleapiclient import discovery
import httplib2
import bq_flags
import bq_utils
from clients import bigquery_http
from clients import utils as bq_client_utils
from discovery_documents import discovery_document_cache
from discovery_documents import discovery_document_loader
from utils import bq_api_utils
from utils import bq_error
def GetAuthorizedHttp(self, credentials: Any, http: Any, is_for_discovery: bool=False):
    """Returns an http client that is authorized with the given credentials."""
    if is_for_discovery:
        credentials = bq_utils.GetSanitizedCredentialForDiscoveryRequest(self.use_google_auth, credentials)
    if self.use_google_auth:
        if not _HAS_GOOGLE_AUTH:
            logging.error('System is set to use `google.auth`, but it did not load.')
        if not isinstance(credentials, google_credentials.Credentials):
            logging.error('The system is using `google.auth` but the parsed credentials are of an incorrect type.')
    else:
        logging.debug('System is set to not use `google.auth`.')
    if _HAS_GOOGLE_AUTH and isinstance(credentials, google_credentials.Credentials):
        if google_auth_httplib2 is None:
            raise ValueError('Credentials from google.auth specified, but google-api-python-client is unable to use these credentials unless google-auth-httplib2 is installed. Please install google-auth-httplib2.')
        return google_auth_httplib2.AuthorizedHttp(credentials, http=http)
    return credentials.authorize(http)