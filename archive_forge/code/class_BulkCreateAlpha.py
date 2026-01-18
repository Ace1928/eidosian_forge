from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import filter_rewrite
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.disks import flags as disks_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class BulkCreateAlpha(BulkCreate):
    """Create multiple Compute Engine disks."""

    @classmethod
    def Args(cls, parser):
        _CommonArgs(parser)

    def Run(self, args):
        return self._Run(args)