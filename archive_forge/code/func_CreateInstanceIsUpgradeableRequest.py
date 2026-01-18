from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.notebooks import environments as env_util
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateInstanceIsUpgradeableRequest(args, messages):
    instance = GetInstanceResource(args).RelativeName()
    return messages.NotebooksProjectsLocationsInstancesIsUpgradeableRequest(notebookInstance=instance)