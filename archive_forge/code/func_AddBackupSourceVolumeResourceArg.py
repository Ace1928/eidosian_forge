from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddBackupSourceVolumeResourceArg(parser, required=True):
    group_help = "The full name of the Source Volume that the Backup is based on',\n      Format: `projects/{projects_id}/locations/{location}/volumes/{volume_id}`\n      "
    concept_parsers.ConceptParser.ForResource('--source-volume', flags.GetVolumeResourceSpec(positional=False), group_help=group_help, required=required, flag_name_overrides={'location': ''}).AddToParser(parser)