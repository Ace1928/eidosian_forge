from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
@staticmethod
def _MakeIgmPatchResource(client, args):
    igm_patch_resource = client.messages.InstanceGroupManager()
    if args.size is not None:
        igm_patch_resource.targetSize = args.size
    if args.suspended_size is not None:
        igm_patch_resource.targetSuspendedSize = args.suspended_size
    if args.stopped_size is not None:
        igm_patch_resource.targetStoppedSize = args.stopped_size
    return igm_patch_resource