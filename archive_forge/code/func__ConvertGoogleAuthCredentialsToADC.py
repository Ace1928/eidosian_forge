from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import base64
import collections
import copy
import datetime
import enum
import hashlib
import json
import os
import sqlite3
from google.auth import compute_engine as google_auth_compute_engine
from google.auth import credentials as google_auth_creds
from google.auth import exceptions as google_auth_exceptions
from google.auth import external_account as google_auth_external_account
from google.auth import external_account_authorized_user as google_auth_external_account_authorized_user
from google.auth import impersonated_credentials as google_auth_impersonated
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import devshell as c_devshell
from googlecloudsdk.core.credentials import exceptions as c_exceptions
from googlecloudsdk.core.credentials import introspect as c_introspect
from googlecloudsdk.core.util import files
from oauth2client import client
from oauth2client import service_account
from oauth2client.contrib import gce as oauth2client_gce
import six
def _ConvertGoogleAuthCredentialsToADC(credentials):
    """Converts a google-auth credentials to application default credentials."""
    creds_type = CredentialTypeGoogleAuth.FromCredentials(credentials)
    if creds_type == CredentialTypeGoogleAuth.USER_ACCOUNT:
        adc = credentials.to_json(strip=('token', 'token_uri', 'scopes', 'expiry'))
        adc = json.loads(adc)
        adc['type'] = creds_type.key
        return adc
    if creds_type == CredentialTypeGoogleAuth.SERVICE_ACCOUNT:
        return {'type': creds_type.key, 'client_email': credentials.service_account_email, 'private_key_id': credentials.private_key_id, 'private_key': credentials.private_key, 'client_id': credentials.client_id, 'token_uri': credentials._token_uri, 'universe_domain': credentials.universe_domain}
    if creds_type == CredentialTypeGoogleAuth.EXTERNAL_ACCOUNT or creds_type == CredentialTypeGoogleAuth.EXTERNAL_ACCOUNT_USER:
        adc_json = credentials.info
        adc_json.pop('client_id', None)
        adc_json.pop('client_secret', None)
        return adc_json
    if creds_type == CredentialTypeGoogleAuth.EXTERNAL_ACCOUNT_AUTHORIZED_USER:
        adc_json = credentials.to_json(strip=('token', 'expiry', 'scopes'))
        adc_json = json.loads(adc_json)
        if getattr(credentials, 'universe_domain', None) is not None:
            adc_json['universe_domain'] = credentials.universe_domain
        return adc_json
    raise ADCError('Cannot convert credentials of type {} to application default credentials.'.format(type(credentials)))