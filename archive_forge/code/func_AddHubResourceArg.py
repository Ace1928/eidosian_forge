from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.network_connectivity import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddHubResourceArg(parser, desc):
    """Add a resource argument for a hub.

  Args:
    parser: the parser for the command.
    desc: the string to describe the resource, such as 'to create'.
  """
    hub_concept_spec = concepts.ResourceSpec('networkconnectivity.projects.locations.global.hubs', resource_name='hub', hubsId=HubAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)
    presentation_spec = presentation_specs.ResourcePresentationSpec(name='hub', concept_spec=hub_concept_spec, required=True, group_help='Name of the hub {}.'.format(desc))
    concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)