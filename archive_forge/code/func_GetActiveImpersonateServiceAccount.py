from __future__ import absolute_import
from __future__ import unicode_literals
import gcloud
import sys
import json
import os
import platform
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from six.moves import input
def GetActiveImpersonateServiceAccount():
    """Get the active impersonate_service_account property.

  For use with wrapping legacy tools that take impersonate_service_account on
  the command line.

  Returns:
    str, The name of the service account to impersonate.
  """
    return properties.VALUES.auth.impersonate_service_account.Get(validate=False)