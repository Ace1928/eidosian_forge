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
def CredentialsFromFile(path, client_info, oauth2client_args=None):
    """Read credentials from a file."""
    user_agent = client_info['user_agent']
    scope_key = client_info['scope']
    if not isinstance(scope_key, six.string_types):
        scope_key = ':'.join(scope_key)
    storage_key = client_info['client_id'] + user_agent + scope_key
    if _NEW_FILESTORE:
        credential_store = multiprocess_file_storage.MultiprocessFileStorage(path, storage_key)
    else:
        credential_store = multistore_file.get_credential_storage_custom_string_key(path, storage_key)
    if hasattr(FLAGS, 'auth_local_webserver'):
        FLAGS.auth_local_webserver = False
    credentials = credential_store.get()
    if credentials is None or credentials.invalid:
        print('Generating new OAuth credentials ...')
        for _ in range(20):
            try:
                flow = oauth2client.client.OAuth2WebServerFlow(**client_info)
                flags = _GetRunFlowFlags(args=oauth2client_args)
                credentials = tools.run_flow(flow, credential_store, flags)
                break
            except (oauth2client.client.FlowExchangeError, SystemExit) as e:
                print('Invalid authorization: %s' % (e,))
            except httplib2.HttpLib2Error as e:
                print('Communication error: %s' % (e,))
                raise exceptions.CredentialsError('Communication error creating credentials: %s' % e)
    return credentials