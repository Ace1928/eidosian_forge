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
@_RegisterCredentialsMethod
def _GetServiceAccountCredentials(client_info, service_account_name=None, service_account_keyfile=None, service_account_json_keyfile=None, **unused_kwds):
    """Returns ServiceAccountCredentials from give file."""
    scopes = client_info['scope'].split()
    user_agent = client_info['user_agent']
    if service_account_json_keyfile:
        return ServiceAccountCredentialsFromFile(service_account_json_keyfile, scopes, user_agent=user_agent)
    if service_account_name and (not service_account_keyfile) or (service_account_keyfile and (not service_account_name)):
        raise exceptions.CredentialsError('Service account name or keyfile provided without the other')
    if service_account_name is not None:
        return ServiceAccountCredentialsFromP12File(service_account_name, service_account_keyfile, scopes, user_agent)