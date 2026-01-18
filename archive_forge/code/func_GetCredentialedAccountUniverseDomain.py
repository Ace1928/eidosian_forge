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
def GetCredentialedAccountUniverseDomain(account: str) -> Optional[str]:
    """Get the universe domain of a credentialed account.

  Args:
    account: The account to get the universe domain for.

  Returns:
    The credentialed account's universe domain if exists. None otherwise.
  """
    all_cred_accounts = AllAccountsWithUniverseDomains()
    cred_account = next((cred_account for cred_account in all_cred_accounts if cred_account.account == account), None)
    return cred_account.universe_domain if cred_account else None