from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.workbench import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateInstanceStartRequest(args, messages):
    instance = GetInstanceResource(args).RelativeName()
    start_request = messages.StartInstanceRequest()
    return messages.NotebooksProjectsLocationsInstancesStartRequest(name=instance, startInstanceRequest=start_request)