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
def ExportRevisionStateFile(messages, deployment_full_name):
    """Creates a signed uri to download the state file.

  Args:
    messages: ModuleType, the messages module that lets us form Infra Manager
      API messages based on our protos.
    deployment_full_name: string, the fully qualified name of the deployment,
      e.g. "projects/p/locations/l/deployments/d".

  Returns:
    A messages.StateFile which contains signed uri to be used to upload a state
    file.
  """
    export_revision_state_file_request = messages.ExportRevisionStatefileRequest()
    log.status.Print('Initiating export state file request...')
    state_file = configmanager_util.ExportRevisionStateFile(export_revision_state_file_request, deployment_full_name)
    return state_file