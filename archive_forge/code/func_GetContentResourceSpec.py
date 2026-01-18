from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.immersive_stream.xr import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def GetContentResourceSpec():
    return concepts.ResourceSpec(resource_collection='stream.projects.locations.streamContents', api_version='v1alpha1', resource_name='content', streamContentsId=ContentAttributeConfig(), locationsId=LocationAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)