from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.infra_manager import deterministic_snapshot
from googlecloudsdk.command_lib.infra_manager import errors
from googlecloudsdk.command_lib.infra_manager import staging_bucket_util
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def ImportStateFile(messages, deployment_full_name, lock_id, file=None):
    """Creates a signed uri to upload the state file.

  Args:
    messages: ModuleType, the messages module that lets us form Infra Manager
      API messages based on our protos.
    deployment_full_name: string, the fully qualified name of the deployment,
      e.g. "projects/p/locations/l/deployments/d".
    lock_id: Lock ID of the lock file to verify person importing owns lock.
    file: file path of the statefile to be uploaded.

  Returns:
    A messages.StateFile which contains signed uri to be used to upload a state
    file.
  """
    import_state_file_request = messages.ImportStatefileRequest(lockId=int(lock_id))
    log.status.Print('Creating signed uri for ImportStatefile request...')
    state_file = configmanager_util.ImportStateFile(import_state_file_request, deployment_full_name)
    if file is None:
        return state_file
    state_file_url = state_file.signedUri
    state_file_obj = files.BinaryFileReader(os.path.abspath(file))
    requests.GetSession().put(state_file_url, data=state_file_obj)
    log.status.Print(f'Statefile {file} uploaded successfully.')
    return