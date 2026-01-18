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
def DumpExtendedADCToFile(self, file_path=None, quota_project=None):
    """Dumps the credentials and the quota project to the ADC json file."""
    if not self.is_user:
        raise CredentialFileSaveError('The credential is not a user credential, so we cannot insert a quota project to application default credential.')
    file_path = file_path or config.ADCFilePath()
    if not quota_project:
        quota_project = GetQuotaProject(self._credentials, force_resource_quota=True)
    extended_adc = self._ExtendADCWithQuotaProject(quota_project)
    return _DumpADCJsonToFile(extended_adc, file_path)