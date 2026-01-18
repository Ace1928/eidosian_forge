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
def ExportAutomationSelector(manifest, resource_selector):
    """Exports the selector field of the Automation resource.

  Args:
    manifest: A dictionary that represents the cloud deploy Automation resource.
    resource_selector:
      googlecloudsdk.generated_clients.apis.clouddeploy.AutomationResourceSelector
      message.
  """
    manifest[SELECTOR_FIELD] = []
    for selector in getattr(resource_selector, 'targets'):
        manifest[SELECTOR_FIELD].append({TARGET_FIELD: selector})