from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import base
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.credentials import creds as core_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.util import files
from oauth2client import client
import six
from google.auth import exceptions as google_auth_exceptions
def _HandleAuthError(e):
    """Handle a generic auth error and raise a nicer message.

  Args:
    e: The exception that was caught.

  Raises:
    creds_exceptions.TokenRefreshError: If an auth error occurs.
  """
    msg = six.text_type(e)
    log.debug('Exception caught during HTTP request: %s', msg, exc_info=True)
    if context_aware.IsContextAwareAccessDeniedError(e):
        raise creds_exceptions.TokenRefreshDeniedByCAAError(msg)
    raise creds_exceptions.TokenRefreshError(msg)