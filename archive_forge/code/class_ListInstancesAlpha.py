from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import instance_groups_utils
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class ListInstancesAlpha(ListInstances):
    """List Compute Engine instances present in managed instance group."""

    @staticmethod
    def Args(parser):
        instance_groups_flags.AddListInstancesOutputFormat(parser, release_track=base.ReleaseTrack.ALPHA)
        parser.display_info.AddUriFunc(instance_groups_utils.UriFuncForListInstanceRelatedObjects)
        instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.AddArgument(parser)