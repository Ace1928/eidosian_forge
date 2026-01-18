from __future__ import print_function
import argparse
import contextlib
import datetime
import json
import os
import threading
import warnings
import httplib2
import oauth2client
import oauth2client.client
from oauth2client import service_account
from oauth2client import tools  # for gflags declarations
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.py import exceptions
from apitools.base.py import util
def ServiceAccountCredentialsFromFile(filename, scopes, user_agent=None):
    """Use the credentials in filename to create a token for scopes."""
    filename = os.path.expanduser(filename)
    if oauth2client.__version__ > '1.5.2':
        credentials = service_account.ServiceAccountCredentials.from_json_keyfile_name(filename, scopes=scopes)
        if credentials is not None:
            if user_agent is not None:
                credentials.user_agent = user_agent
        return credentials
    else:
        with open(filename) as keyfile:
            service_account_info = json.load(keyfile)
        account_type = service_account_info.get('type')
        if account_type != oauth2client.client.SERVICE_ACCOUNT:
            raise exceptions.CredentialsError('Invalid service account credentials: %s' % (filename,))
        credentials = service_account._ServiceAccountCredentials(service_account_id=service_account_info['client_id'], service_account_email=service_account_info['client_email'], private_key_id=service_account_info['private_key_id'], private_key_pkcs8_text=service_account_info['private_key'], scopes=scopes, user_agent=user_agent)
        return credentials