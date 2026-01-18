from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from google.auth import external_account as google_auth_external_account
import google_auth_httplib2
from googlecloudsdk.calliope import base
from googlecloudsdk.core import http
from googlecloudsdk.core.credentials import creds as core_creds
from googlecloudsdk.core.credentials import transport
import six
class _GoogleAuthApitoolsCredentials:
    """Class of wrapping credentials."""

    def __init__(self, credentials):
        self.credentials = credentials

    def refresh(self, http_client):
        del http_client
        if isinstance(self.credentials, google_auth_external_account.Credentials) and self.credentials.valid:
            return
        self.credentials.refresh(http.GoogleAuthRequest())