from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.deploy import automation_util
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def SetCustomTarget(target, custom_target, project, region):
    """Sets the customTarget field of cloud deploy target message.

  This is handled specially because we allow providing either the ID or name for
  the custom target type referenced. When the ID is provided we need to
  construct the name.

  Args:
    target: googlecloudsdk.generated_clients.apis.clouddeploy.Target message.
    custom_target:
      googlecloudsdk.generated_clients.apis.clouddeploy.CustomTarget message.
    project: str, gcp project.
    region: str, ID of the location.
  """
    custom_target_type = custom_target.get('customTargetType')
    if '/' in custom_target_type:
        target.customTarget = custom_target
        return
    custom_target_type_resource_ref = resources.REGISTRY.Parse(None, collection='clouddeploy.projects.locations.customTargetTypes', params={'projectsId': project, 'locationsId': region, 'customTargetTypesId': custom_target_type})
    custom_target['customTargetType'] = custom_target_type_resource_ref.RelativeName()
    target.customTarget = custom_target