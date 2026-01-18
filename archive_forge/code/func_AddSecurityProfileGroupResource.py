from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.network_security.security_profile_groups import spg_api
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import resources
def AddSecurityProfileGroupResource(parser, release_track):
    """Adds Security Profile Group."""
    name = _SECURITY_PROFILE_GROUP_RESOURCE_NAME
    resource_spec = concepts.ResourceSpec(resource_collection=_SECURITY_PROFILE_GROUP_RESOURCE_COLLECTION, resource_name='security_profile_group', api_version=spg_api.GetApiVersion(release_track), organizationsId=concepts.ResourceParameterAttributeConfig('organization', 'Organization ID of Security Profile Group', parameter_name='organizationsId'), locationsId=concepts.ResourceParameterAttributeConfig('location', 'location of the {resource} - Global.', parameter_name='locationsId'), securityProfileGroupsId=concepts.ResourceParameterAttributeConfig('security_profile_group', 'Name of security profile group {resource}.', parameter_name='securityProfileGroupsId'))
    presentation_spec = presentation_specs.ResourcePresentationSpec(name=name, concept_spec=resource_spec, required=True, group_help='Security Profile Group Name.')
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)