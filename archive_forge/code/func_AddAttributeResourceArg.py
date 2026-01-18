from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddAttributeResourceArg(parser, verb, positional=True):
    """Adds a resource argument for a Dataplex Attribute."""
    name = 'data_attribute' if positional else '--data_attribute'
    return concept_parsers.ConceptParser.ForResource(name, GetDataAttributeResourceSpec(), 'The DataAttribute {}'.format(verb), required=True).AddToParser(parser)