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
def ParseImpersonationAccounts(service_account_ids):
    """Finds the target impersonation principal and the delegates.

  Args:
     service_account_ids: str, A list of service account ids separated using
       comma.

  Returns:
     A tuple (target_principal, delegates).

  Raises:
    NoImpersonationAccountError: if the input does not contain service accounts.
  """
    service_account_ids = service_account_ids.split(',')
    service_account_ids = [sa_id.strip() for sa_id in service_account_ids]
    if not service_account_ids:
        raise NoImpersonationAccountError('No service account to impersonate.')
    return (service_account_ids[-1], service_account_ids[:-1] or None)