from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddBackupVaultUpdateArgs(parser):
    """Add args for updating a Backup Vault."""
    concept_parsers.ConceptParser([flags.GetBackupVaultPresentationSpec('The Backup Vault to update')]).AddToParser(parser)
    flags.AddResourceDescriptionArg(parser, 'Backup Vault')
    flags.AddResourceAsyncFlag(parser)
    labels_util.AddUpdateLabelsFlags(parser)