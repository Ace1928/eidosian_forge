from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddBackupVaultDeleteArgs(parser):
    """Add args for deleting a Backup Vault."""
    concept_parsers.ConceptParser([flags.GetBackupVaultPresentationSpec('The Backup Vault to delete')]).AddToParser(parser)
    flags.AddResourceAsyncFlag(parser)