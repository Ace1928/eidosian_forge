from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instant_snapshots import flags as ips_flags
from googlecloudsdk.command_lib.util.args import labels_util
import six
def _GetSourceDiskUri(self, args, compute_holder, default_scope):
    source_disk_ref = ips_flags.SOURCE_DISK_ARG.ResolveAsResource(args, compute_holder.resources)
    if source_disk_ref:
        return source_disk_ref.SelfLink()
    return None