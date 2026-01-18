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
def AddEndpointResource(release_track, parser):
    """Adds Firewall Plus endpoint resource."""
    api_version = activation_api.GetApiVersion(release_track)
    collection_info = resources.REGISTRY.Clone().GetCollectionInfo(ASSOCIATION_RESOURCE_COLLECTION, api_version)
    resource_spec = concepts.ResourceSpec(ENDPOINT_RESOURCE_COLLECTION, 'firewall endpoint', api_version=api_version, organizationsId=concepts.ResourceParameterAttributeConfig('organization', 'Organization ID to which the changes should apply.', parameter_name='organizationsId'), locationsId=concepts.ResourceParameterAttributeConfig('endpoint-zone', 'Zone of the {resource}.', parameter_name='locationsId', fallthroughs=[deps.ArgFallthrough('--zone'), deps.FullySpecifiedAnchorFallthrough([deps.ArgFallthrough(ASSOCIATION_RESOURCE_NAME)], collection_info, 'locationsId')]), firewallEndpointsId=concepts.ResourceParameterAttributeConfig('endpoint-name', 'Name of the {resource}', parameter_name='firewallEndpointsId'))
    presentation_spec = presentation_specs.ResourcePresentationSpec(name='--endpoint', concept_spec=resource_spec, required=True, group_help='Firewall Plus.')
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)