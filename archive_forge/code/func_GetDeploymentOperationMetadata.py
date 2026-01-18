from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from typing import List, Optional
from apitools.base.py import encoding as apitools_encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as api_lib_exceptions
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import retry
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_client
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def GetDeploymentOperationMetadata(messages, operation: runapps_v1alpha1_messages.Operation) -> runapps_v1alpha1_messages.DeploymentOperationMetadata:
    """Get the metadata message for the deployment operation.

  Args:
    messages: Module containing the definitions of messages for the Runapps
      API.
    operation: The LRO

  Returns:
    The DeploymentOperationMetadata object.
  """
    return apitools_encoding.PyValueToMessage(messages.DeploymentOperationMetadata, apitools_encoding.MessageToPyValue(operation.metadata))