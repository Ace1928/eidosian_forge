from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from google.auth import external_account as google_auth_external_account
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import requests
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import store
class MissingStoredCredentialsError(exceptions.Error):
    """Indicates stored credentials do not exist or are not available."""