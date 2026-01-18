import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
def _get_application_default_credential_from_file(filename):
    """Build the Application Default Credentials from file."""
    with open(filename) as file_obj:
        client_credentials = json.load(file_obj)
    credentials_type = client_credentials.get('type')
    if credentials_type == AUTHORIZED_USER:
        required_fields = set(['client_id', 'client_secret', 'refresh_token'])
    elif credentials_type == SERVICE_ACCOUNT:
        required_fields = set(['client_id', 'client_email', 'private_key_id', 'private_key'])
    else:
        raise ApplicationDefaultCredentialsError("'type' field should be defined (and have one of the '" + AUTHORIZED_USER + "' or '" + SERVICE_ACCOUNT + "' values)")
    missing_fields = required_fields.difference(client_credentials.keys())
    if missing_fields:
        _raise_exception_for_missing_fields(missing_fields)
    if client_credentials['type'] == AUTHORIZED_USER:
        return GoogleCredentials(access_token=None, client_id=client_credentials['client_id'], client_secret=client_credentials['client_secret'], refresh_token=client_credentials['refresh_token'], token_expiry=None, token_uri=oauth2client.GOOGLE_TOKEN_URI, user_agent='Python client library')
    else:
        from oauth2client import service_account
        return service_account._JWTAccessCredentials.from_json_keyfile_dict(client_credentials)