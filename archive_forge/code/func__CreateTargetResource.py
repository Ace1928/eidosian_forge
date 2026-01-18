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
def _CreateTargetResource(messages, target_name_or_id, project, region):
    """Creates target resource with full target name and the resource reference."""
    resource = messages.Target()
    resource_ref = target_util.TargetReference(target_name_or_id, project, region)
    resource.name = resource_ref.RelativeName()
    return (resource, resource_ref)