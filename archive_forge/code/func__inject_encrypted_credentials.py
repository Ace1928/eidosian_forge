from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import files
def _inject_encrypted_credentials(dataproc, session_id, credentials_ciphertext, request_id=None):
    """Inject credentials into session.

  The credentials must have already been encrypted before calling this
  method.
  Args:
    dataproc: The API client for calling into the Dataproc API.
    session_id: Relative name of the session. Format:
      'projects/{}/locations/{}/session/{}'
    credentials_ciphertext: The (already encrypted) credentials to inject.
    request_id: (optional) A unique ID used to identify the inject credentials
      request.

  Returns:
    An operation resource for the credential injection.
  """
    inject_session_credentials_request = dataproc.messages.InjectSessionCredentialsRequest(credentialsCiphertext=credentials_ciphertext, requestId=request_id)
    request = dataproc.messages.DataprocProjectsLocationsSessionsInjectCredentialsRequest(injectSessionCredentialsRequest=inject_session_credentials_request, session=session_id)
    return dataproc.client.projects_locations_sessions.InjectCredentials(request)