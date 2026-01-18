from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def _ConstructClusterResourceForRestoreRequestAlpha(alloydb_messages, args):
    """Returns the cluster resource for restore request."""
    cluster_resource = _ConstructClusterResourceForRestoreRequest(alloydb_messages, args)
    return cluster_resource