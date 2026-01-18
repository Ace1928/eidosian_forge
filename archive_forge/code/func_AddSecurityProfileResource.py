from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.network_security.security_profiles.threat_prevention import sp_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddSecurityProfileResource(parser, release_track):
    """Adds Security Profile Threat Prevention type."""
    name = 'security_profile'
    resource_spec = concepts.ResourceSpec(resource_collection='networksecurity.organizations.locations.securityProfiles', resource_name='security_profile', api_version=sp_api.GetApiVersion(release_track), organizationsId=concepts.ResourceParameterAttributeConfig('organization', 'Organization ID to which the changes should apply.', parameter_name='organizationsId'), locationsId=concepts.ResourceParameterAttributeConfig('location', 'location of the {resource} - Global.', parameter_name='locationsId'), securityProfilesId=concepts.ResourceParameterAttributeConfig('security_profile', 'Name of the {resource}.', parameter_name='securityProfilesId'))
    presentation_spec = presentation_specs.ResourcePresentationSpec(name=name, concept_spec=resource_spec, required=True, group_help='Security Profile Name.')
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)