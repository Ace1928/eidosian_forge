from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddBackupUpdateArgs(parser):
    """Add args for updating a Backup."""
    concept_parsers.ConceptParser([flags.GetBackupPresentationSpec('The Backup to update')]).AddToParser(parser)
    AddBackupBackupVaultResourceArg(parser, required=True)
    flags.AddResourceDescriptionArg(parser, 'Backup')
    flags.AddResourceAsyncFlag(parser)
    labels_util.AddUpdateLabelsFlags(parser)