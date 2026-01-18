from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.notebooks import environments as env_util
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateInstanceRegisterRequest(args, messages):
    instance = GetInstanceResource(args)
    parent = util.GetLocationResource(instance.locationsId, instance.projectsId).RelativeName()
    register_request = messages.RegisterInstanceRequest(instanceId=instance.Name())
    return messages.NotebooksProjectsLocationsInstancesRegisterRequest(parent=parent, registerInstanceRequest=register_request)