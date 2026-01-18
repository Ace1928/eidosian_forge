from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddRegionResourceArg(parser, verb):
    """Add a resource argument for a Vertex AI region.

  NOTE: Must be used only if it's the only resource arg in the command.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
  """
    region_resource_spec = concepts.ResourceSpec('datapipelines.projects.locations', resource_name='region', locationsId=RegionAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)
    concept_parsers.ConceptParser.ForResource('--region', region_resource_spec, 'Cloud region {}.'.format(verb), required=True).AddToParser(parser)