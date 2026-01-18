from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def _ConstructClusterForCreateRequestAlpha(alloydb_messages, args):
    """Returns the cluster for alpha create request based on args."""
    flags.ValidateConnectivityFlags(args)
    cluster = _ConstructClusterForCreateRequestBeta(alloydb_messages, args)
    return cluster