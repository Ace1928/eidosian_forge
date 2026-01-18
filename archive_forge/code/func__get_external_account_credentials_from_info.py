import copy
import datetime
import io
import json
from google.auth import aws
from google.auth import credentials
from google.auth import exceptions
from google.auth import external_account
from google.auth import external_account_authorized_user
from google.auth import identity_pool
from google.auth import pluggable
from google.auth.transport import requests
from gslib.utils import constants
import oauth2client
def _get_external_account_credentials_from_info(info):
    if info.get('subject_token_type') == 'urn:ietf:params:aws:token-type:aws4_request':
        return aws.Credentials.from_info(info, scopes=DEFAULT_SCOPES)
    elif info.get('credential_source') is not None and info.get('credential_source').get('executable') is not None:
        return pluggable.Credentials.from_info(info, scopes=DEFAULT_SCOPES)
    else:
        return identity_pool.Credentials.from_info(info, scopes=DEFAULT_SCOPES)