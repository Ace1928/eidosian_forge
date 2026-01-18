from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddEntryTypeResourceArg(parser):
    """Adds a resource argument for a Dataplex EntryType."""
    return concept_parsers.ConceptParser.ForResource('--entry-type', GetEntryTypeResourceSpec(), 'Arguments and flags that define the Dataplex EntryType you want to reference.', required=True).AddToParser(parser)