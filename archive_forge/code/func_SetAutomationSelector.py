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
def SetAutomationSelector(messages, automation, selectors):
    """Sets the selectors field of cloud deploy automation resource message.

  Args:
    messages: module containing the definitions of messages for Cloud Deploy.
    automation:  googlecloudsdk.generated_clients.apis.clouddeploy.Automation
      message.
    selectors:
      [googlecloudsdk.generated_clients.apis.clouddeploy.TargetAttributes], list
      of TargetAttributes messages.
  """
    automation.selector = messages.AutomationResourceSelector()
    if not isinstance(selectors, list):
        for target_attribute in selectors.get(TARGETS_FIELD):
            _AddTargetAttribute(messages, automation.selector, target_attribute)
    else:
        for selector in selectors:
            message = selector.get(TARGET_FIELD)
            _AddTargetAttribute(messages, automation.selector, message)