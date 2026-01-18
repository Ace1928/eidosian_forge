from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddDataplexAspectTypeResourceArg(parser, verb, positional=True):
    """Adds a resource argument for a Dataplex AspectType."""
    name = 'aspect_type' if positional else '--aspect_type'
    return concept_parsers.ConceptParser.ForResource(name, GetDataplexAspectTypeResourceSpec(), 'Arguments and flags that define the Dataplex aspect type you want {}'.format(verb), required=True).AddToParser(parser)