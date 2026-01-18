from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import p12_service_account
from googlecloudsdk.core.util import files
from oauth2client import service_account
def IsServiceAccountConfig(content_json):
    """Returns whether a JSON content corresponds to an service account cred."""
    return (content_json or {}).get('type') == _SERVICE_ACCOUNT_TYPE