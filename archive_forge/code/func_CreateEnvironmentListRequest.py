from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateEnvironmentListRequest(args, messages):
    parent = util.GetParentFromArgs(args)
    return messages.NotebooksProjectsLocationsEnvironmentsListRequest(parent=parent)