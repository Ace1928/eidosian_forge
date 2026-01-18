from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetImageResourceSpec():
    return concepts.ResourceSpec('compute.images', resource_name='image', project=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)