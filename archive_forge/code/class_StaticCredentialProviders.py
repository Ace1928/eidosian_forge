from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import datetime
import json
import os
import textwrap
import time
from typing import Optional
import dateutil
from googlecloudsdk.api_lib.auth import util as auth_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import gce as c_gce
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
from oauth2client import client
from oauth2client import crypt
from oauth2client import service_account
from oauth2client.contrib import gce as oauth2client_gce
from oauth2client.contrib import reauth_errors
import six
from six.moves import urllib
class StaticCredentialProviders(object):
    """Manages a list of credential providers."""

    def __init__(self):
        self._providers = []

    def AddProvider(self, provider):
        self._providers.append(provider)

    def RemoveProvider(self, provider):
        self._providers.remove(provider)

    def GetCredentials(self, account, use_google_auth=True):
        for provider in self._providers:
            cred = provider.GetCredentials(account, use_google_auth)
            if cred is not None:
                return cred
        return None

    def GetAccounts(self):
        accounts = set()
        for provider in self._providers:
            accounts |= provider.GetAccounts()
        return accounts