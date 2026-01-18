from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.iam import util as iam_api
from googlecloudsdk.api_lib.resource_manager import tags
from googlecloudsdk.api_lib.resource_manager.exceptions import ResourceManagerError
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.resource_manager import endpoint_utils as endpoints
from googlecloudsdk.core import exceptions as core_exceptions
def ProjectNameToBinding(project_name, tag_value, location=None):
    """Returns the binding name given a project name and tag value.

  Requires binding list permission.

  Args:
    project_name: project name provided, fully qualified resource name
    tag_value: tag value to match the binding name to
    location: region or zone

  Returns:
    binding_name

  Raises:
    InvalidInputError: project not found
  """
    service = Services[TAG_BINDINGS]()
    with endpoints.CrmEndpointOverrides(location):
        req = ListRequests[TAG_BINDINGS](parent=project_name)
        response = service.List(req)
        for bn in response.tagBindings:
            if bn.tagValue == tag_value:
                return bn.name
        raise InvalidInputError('Binding not found for parent [{}], tagValue [{}]'.format(project_name, tag_value))