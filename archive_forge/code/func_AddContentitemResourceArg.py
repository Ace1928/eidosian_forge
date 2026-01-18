from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddContentitemResourceArg(parser, verb, positional=True):
    """Adds a resource argument for a Dataplex Content."""
    name = 'content' if positional else '--content'
    return concept_parsers.ConceptParser.ForResource(name, GetContentitemResourceSpec(), 'The Content {}'.format(verb), required=True).AddToParser(parser)