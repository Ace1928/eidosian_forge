from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.binauthz import arg_parsers
from googlecloudsdk.command_lib.kms import flags as kms_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs as presentation_specs_lib
def _GetPlatformPolicyResourceSpec():
    return concepts.ResourceSpec('binaryauthorization.projects.platforms.policies', resource_name='policy', api_version='v1', projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, platformsId=concepts.ResourceParameterAttributeConfig(name='platform', help_text='The platform that the {resource} belongs to. PLATFORM must be one of the following: cloudRun, gke.'), policyId=concepts.ResourceParameterAttributeConfig(name='policy', help_text='The ID of the {resource}.'))