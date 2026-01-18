from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.redis import cluster_util
from googlecloudsdk.command_lib.redis import util
def UpdateShardCount(unused_cluster_ref, args, patch_request):
    """Hook to add shard count to the redis cluster update request."""
    if args.IsSpecified('shard_count'):
        patch_request.cluster.shardCount = args.shard_count
        patch_request = AddFieldToUpdateMask('shard_count', patch_request)
    return patch_request