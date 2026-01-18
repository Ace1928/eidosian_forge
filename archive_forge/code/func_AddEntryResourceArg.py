from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddEntryResourceArg(parser):
    """Adds a resource argument for a Dataplex Entry."""
    entry_data = yaml_data.ResourceYAMLData.FromPath('dataplex.entry')
    return concept_parsers.ConceptParser.ForResource('entry', concepts.ResourceSpec.FromYaml(entry_data.GetData(), is_positional=True), 'Arguments and flags that define the Dataplex Entry you want to reference.', required=True).AddToParser(parser)