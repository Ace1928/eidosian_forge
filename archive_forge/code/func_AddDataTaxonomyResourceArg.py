from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddDataTaxonomyResourceArg(parser, verb, positional=True):
    """Adds a resource argument for a Dataplex Data Taxonomy."""
    name = 'data_taxonomy' if positional else '--data_taxonomy'
    return concept_parsers.ConceptParser.ForResource(name, GetDataTaxonomyResourceSpec(), 'The DataTaxonomy {}'.format(verb), required=True).AddToParser(parser)