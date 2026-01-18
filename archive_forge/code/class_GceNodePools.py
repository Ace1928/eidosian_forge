from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class GceNodePools(base.Group):
    """Manage Dataproc node groups.

  Manage and modify Dataproc node groups created with a parent cluster.

  ## EXAMPLES

  To describe a node group:

    $ {command} describe NODE_GROUP_ID --cluster cluster_name --region region

  To resize a node group:

    $ {command} resize NODE_GROUP_ID --cluster cluster_name --region region
    --size new_size
  """