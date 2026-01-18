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
def AddVolumeDailySnapshotArg(parser):
    """Adds the --snapshot-daily arg to the arg parser."""
    daily_snapshot_arg_spec = {'snapshots-to-keep': float, 'minute': float, 'hour': float}
    daily_snapshot_help = '\n  Make a snapshot every day e.g. at 06:00, 05:20, 23:50\n  '
    parser.add_argument('--snapshot-daily', type=arg_parsers.ArgDict(spec=daily_snapshot_arg_spec), help=daily_snapshot_help)