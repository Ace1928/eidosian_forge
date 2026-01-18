from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
def AddSnapshotScheduleArgs(parser, messages):
    """Adds flags specific to snapshot schedule resource policies."""
    AddSnapshotMaxRetentionDaysArgs(parser)
    AddOnSourceDiskDeleteArgs(parser, messages)
    snapshot_properties_group = parser.add_group('Snapshot properties')
    AddSnapshotLabelArgs(snapshot_properties_group)
    snapshot_properties_group.add_argument('--guest-flush', action='store_true', help='Create an application consistent snapshot by informing the OS to prepare for the snapshot process.')
    compute_flags.AddStorageLocationFlag(snapshot_properties_group, 'snapshot')