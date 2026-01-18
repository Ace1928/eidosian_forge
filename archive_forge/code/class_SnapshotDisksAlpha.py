from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import name_generator
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.disks import flags as disks_flags
from googlecloudsdk.command_lib.compute.snapshots import flags as snap_flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from six.moves import zip
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class SnapshotDisksAlpha(SnapshotDisksBeta):
    """Create snapshots of Google Compute Engine persistent disks alpha."""

    @classmethod
    def Args(cls, parser):
        SnapshotDisks.disks_arg = disks_flags.MakeDiskArg(plural=True)
        labels_util.AddCreateLabelsFlags(parser)
        _CommonArgs(parser)

    def Run(self, args):
        return self._Run(args)