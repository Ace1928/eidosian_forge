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
def SetDeployParametersForTarget(messages, target, target_ref, deploy_parameters):
    """Sets the deployParameters field of cloud deploy target message.

  Args:
   messages: module containing the definitions of messages for Cloud Deploy.
   target: googlecloudsdk.generated_clients.apis.clouddeploy.Target message.
   target_ref: protorpc.messages.Message, target resource object.
   deploy_parameters: dict[str,str], a dict of deploy parameters (key,value)
     pairs.
  """
    _EnsureIsType(deploy_parameters, dict, 'failed to parse target {}, deployParameters are defined incorrectly'.format(target_ref.Name()))
    dps_message = getattr(messages, deploy_util.ResourceType.TARGET.value).DeployParametersValue
    dps_value = dps_message()
    for key, value in deploy_parameters.items():
        dps_value.additionalProperties.append(dps_message.AdditionalProperty(key=key, value=value))
    target.deployParameters = dps_value