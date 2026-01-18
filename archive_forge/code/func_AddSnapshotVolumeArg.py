from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddSnapshotVolumeArg(parser):
    concept_parsers.ConceptParser.ForResource('--volume', flags.GetVolumeResourceSpec(positional=False), 'The Volume to take a Snapshot of.', flag_name_overrides={'location': ''}).AddToParser(parser)