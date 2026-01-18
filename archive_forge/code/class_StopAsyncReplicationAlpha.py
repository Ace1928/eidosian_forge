from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.disks import flags as disks_flags
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class StopAsyncReplicationAlpha(StopAsyncReplication):
    """Stop Async Replication on Compute Engine persistent disks."""

    @classmethod
    def Args(cls, parser):
        StopAsyncReplication.disks_arg = disks_flags.MakeDiskArg(plural=False)
        _CommonArgs(parser)

    def Run(self, args):
        return self._Run(args)