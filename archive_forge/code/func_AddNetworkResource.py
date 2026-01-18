from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.network_security.firewall_endpoints import activation_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import resources
def AddNetworkResource(parser):
    """Adds network resource."""
    resource_spec = concepts.ResourceSpec('compute.networks', 'network', api_version='v1', project=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, network=concepts.ResourceParameterAttributeConfig('network-name', 'Name of the {resource}', parameter_name='network'))
    presentation_spec = presentation_specs.ResourcePresentationSpec(name='--network', concept_spec=resource_spec, required=True, group_help='Firewall Plus.')
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)