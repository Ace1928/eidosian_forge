from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddVolumeHourlySnapshotArg(parser):
    """Adds the --snapshot-hourly arg to the arg parser."""
    hourly_snapshot_arg_spec = {'snapshots-to-keep': float, 'minute': float}
    hourly_snapshot_help = '\n  Make a snapshot every hour e.g. at 04:00, 05:20, 06:00\n  '
    parser.add_argument('--snapshot-hourly', type=arg_parsers.ArgDict(spec=hourly_snapshot_arg_spec), help=hourly_snapshot_help)