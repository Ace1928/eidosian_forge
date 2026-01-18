from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddSnapshotUpdateArgs(parser):
    """Add args for updating a Snapshot."""
    concept_parsers.ConceptParser([flags.GetSnapshotPresentationSpec('The Snapshot to update.')]).AddToParser(parser)
    AddSnapshotVolumeArg(parser)
    flags.AddResourceAsyncFlag(parser)
    flags.AddResourceDescriptionArg(parser, 'Snapshot')
    labels_util.AddUpdateLabelsFlags(parser)