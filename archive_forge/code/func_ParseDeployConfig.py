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
def ParseDeployConfig(messages, manifests, region):
    """Parses the declarative definition of the resources into message.

  Args:
    messages: module containing the definitions of messages for Cloud Deploy.
    manifests: [str], the list of parsed resource yaml definitions.
    region: str, location ID.

  Returns:
    A dictionary of resource kind and message.
  Raises:
    exceptions.CloudDeployConfigError, if the declarative definition is
    incorrect.
  """
    resource_dict = {DELIVERY_PIPELINE_KIND_V1BETA1: [], TARGET_KIND_V1BETA1: [], AUTOMATION_KIND: [], CUSTOM_TARGET_TYPE_KIND: [], DEPLOY_POLICY_KIND: []}
    project = properties.VALUES.core.project.GetOrFail()
    _ValidateConfig(manifests)
    for manifest in manifests:
        _ParseV1Config(messages, manifest['kind'], manifest, project, region, resource_dict)
    return resource_dict