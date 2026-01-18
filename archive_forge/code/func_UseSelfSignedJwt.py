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
def UseSelfSignedJwt(creds):
    """Check if self signed jwt should be used.

  Only use self signed jwt for google-auth service account creds, and only when
  service_account_use_self_signed_jwt property is true or the universe is not
  the default one.

  Args:
    creds: google.auth.credentials.Credentials, The credentials to check if
      self signed jwt should be used.

  Returns:
    bool, Whether or not self signed jwt should be used.
  """
    cred_type = CredentialTypeGoogleAuth.FromCredentials(creds)
    if cred_type != CredentialTypeGoogleAuth.SERVICE_ACCOUNT:
        return False
    if creds.universe_domain != properties.VALUES.core.universe_domain.default:
        return True
    if properties.VALUES.auth.service_account_use_self_signed_jwt.GetBool():
        return True
    if not properties.IsDefaultUniverse():
        return True
    return False