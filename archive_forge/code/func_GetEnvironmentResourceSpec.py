from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetEnvironmentResourceSpec():
    return concepts.ResourceSpec('composer.projects.locations.environments', resource_name='environment', projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=EnvironmentLocationAttributeConfig(), environmentsId=EnvironmentAttributeConfig())