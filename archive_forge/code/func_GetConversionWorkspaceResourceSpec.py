from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetConversionWorkspaceResourceSpec(resource_name='conversion_workspace'):
    return concepts.ResourceSpec('datamigration.projects.locations.conversionWorkspaces', resource_name=resource_name, api_version='v1', conversionWorkspacesId=ConversionWorkspaceAttributeConfig(name=resource_name), locationsId=RegionAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)