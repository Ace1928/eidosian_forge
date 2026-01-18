from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateEnvironmentCreateRequest(args, messages):
    parent = util.GetParentForEnvironment(args)
    environment = CreateEnvironment(args, messages)
    return messages.NotebooksProjectsLocationsEnvironmentsCreateRequest(parent=parent, environment=environment, environmentId=args.environment)