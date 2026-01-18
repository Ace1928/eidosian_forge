from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddDatascanResourceArg(parser, verb, positional=True):
    """Adds a resource argument for a Dataplex Datascan."""
    name = 'datascan' if positional else '--datascan'
    return concept_parsers.ConceptParser.ForResource(name, GetDatascanResourceSpec(), 'Arguments and flags that define the Dataplex datascan you want {}'.format(verb), required=True).AddToParser(parser)