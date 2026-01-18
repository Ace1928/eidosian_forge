from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
@staticmethod
def _GkeNodePoolTargetFromArgPool(dataproc, gke_cluster, arg_pool, support_shuffle_service=False):
    """Creates a GkeNodePoolTarget from a --pool argument."""
    return dataproc.messages.GkeNodePoolTarget(nodePool='{0}/nodePools/{1}'.format(gke_cluster, arg_pool['name']), roles=_GkeNodePoolTargetParser._SplitRoles(dataproc, arg_pool['roles'], support_shuffle_service), nodePoolConfig=_GkeNodePoolTargetParser._GkeNodePoolConfigFromArgPool(dataproc, arg_pool))