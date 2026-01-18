from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddBackupCreateArgs(parser):
    """Add args for creating a Backup."""
    concept_parsers.ConceptParser([flags.GetBackupPresentationSpec('The Backup to create')]).AddToParser(parser)
    AddBackupBackupVaultResourceArg(parser, required=True)
    AddBackupSourceVolumeResourceArg(parser, required=True)
    AddBackupSourceSnapshotResourceArg(parser)
    flags.AddResourceDescriptionArg(parser, 'Backup Vault')
    flags.AddResourceAsyncFlag(parser)
    labels_util.AddCreateLabelsFlags(parser)