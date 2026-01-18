from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ids import ids_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddOperationResource(parser):
    """Adds Operation resource."""
    name = 'operation'
    resource_spec = concepts.ResourceSpec('ids.projects.locations.operations', 'operation', projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=concepts.ResourceParameterAttributeConfig('zone', 'Zone of the {resource}.', parameter_name='locationsId'), operationsId=concepts.ResourceParameterAttributeConfig('operation', 'Name of the {resource}'))
    presentation_spec = presentation_specs.ResourcePresentationSpec(name=name, concept_spec=resource_spec, required=True, group_help='operation.')
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)