from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import List, Optional, Dict
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.core.console import progress_tracker
def _TypeToDescriptiveName(resource_type):
    """Returns a more readable name for a resource type, for printing to console.

  Args:
    resource_type: type to be described.

  Returns:
    string with readable type name.
  """
    metadata = types_utils.GetTypeMetadataByResourceType(resource_type)
    if metadata and metadata.product:
        return metadata.product
    elif resource_type == 'service':
        return 'Cloud Run Service'
    elif resource_type == 'vpc':
        return 'VPC Connector'
    return resource_type