from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddBackupBackupVaultResourceArg(parser, required=True):
    group_help = 'The Backup Vault that the Backup is stored in'
    concept_parsers.ConceptParser.ForResource('--backup-vault', flags.GetBackupVaultResourceSpec(positional=False), group_help=group_help, required=required, flag_name_overrides={'location': ''}).AddToParser(parser)