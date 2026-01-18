from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.core.resource import resource_transform
import six
def TransformOperationHttpStatus(r, undefined=''):
    """Returns the HTTP response code of an operation.

  Args:
    r: JSON-serializable object.
    undefined: Returns this value if there is no response code.

  Returns:
    The HTTP response code of the operation in r.
  """
    if resource_transform.GetKeyValue(r, 'status', None) == 'DONE':
        return resource_transform.GetKeyValue(r, 'httpErrorStatusCode', None) or 200
    return undefined