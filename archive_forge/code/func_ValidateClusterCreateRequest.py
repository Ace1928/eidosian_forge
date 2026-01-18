from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.edge_cloud.container import admin_users
from googlecloudsdk.command_lib.edge_cloud.container import fleet
from googlecloudsdk.command_lib.edge_cloud.container import resource_args
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.core import resources
def ValidateClusterCreateRequest(req, release_track):
    """Validate cluster create request message.

  Args:
    req: Create cluster request message.
    release_track: Release track of the command.

  Returns:
    Single string of error message.
  """
    messages = util.GetMessagesModule(release_track)
    if req.cluster.releaseChannel == messages.Cluster.ReleaseChannelValueValuesEnum.REGULAR and req.cluster.targetVersion is not None:
        return 'Invalid Argument: REGULAR release channel does not support specification of version'
    return None