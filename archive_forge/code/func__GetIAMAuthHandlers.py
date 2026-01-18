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
def _GetIAMAuthHandlers(authority_selector, authorization_token_file):
    """Get the request handlers for IAM authority selctors and auth tokens..

  Args:
    authority_selector: str, The authority selector string we want to use for
      the request or None.
    authorization_token_file: str, The file that contains the authorization
      token we want to use for the request or None.

  Returns:
    [transport Modifiers]: A list of request modifier functions to use to wrap
    an http request.
  """
    authorization_token = None
    if authorization_token_file:
        try:
            authorization_token = files.ReadFileContents(authorization_token_file)
        except files.Error as e:
            raise Error(e)
    handlers = []
    if authority_selector:
        handlers.append(transport.Handler(transport.SetHeader('x-goog-iam-authority-selector', authority_selector)))
    if authorization_token:
        handlers.append(transport.Handler(transport.SetHeader('x-goog-iam-authorization-token', authorization_token.strip())))
    return handlers