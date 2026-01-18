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
Get credentials and inject them into session.

    Args:
      dataproc: The API client for calling into the Dataproc API.
      session_name: The name of the session.
      session_id: Relative name of the session. Format:
        'projects/{}/locations/{}/session/{}'
      session_key: The public key for the session.
      access_boundary_json: The JSON-formatted access boundary.
      operation_poller: Poller for the cloud operation.
      openssl_executable: The path to the openssl executable.
    