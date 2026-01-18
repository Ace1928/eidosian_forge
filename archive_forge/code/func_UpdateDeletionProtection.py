from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.redis import cluster_util
from googlecloudsdk.command_lib.redis import util
def UpdateDeletionProtection(unused_cluster_ref, args, patch_request):
    """Hook to add delete protection to the redis cluster update request."""
    if args.IsSpecified('deletion_protection'):
        patch_request.cluster.deletionProtectionEnabled = args.deletion_protection
        patch_request = AddFieldToUpdateMask('deletion_protection_enabled', patch_request)
    return patch_request