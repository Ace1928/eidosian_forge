from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddBackupSourceSnapshotResourceArg(parser):
    group_help = "\n      The full name of the Source Snapshot that the Backup is based on',\n      Format: `projects/{project_id}/locations/{location}/volumes/{volume_id}/snapshots/{snapshot_id}`\n      "
    concept_parsers.ConceptParser.ForResource('--source-snapshot', flags.GetSnapshotResourceSpec(source_snapshot_op=True, positional=False), group_help=group_help, flag_name_overrides={'location': '', 'volume': ''}).AddToParser(parser)