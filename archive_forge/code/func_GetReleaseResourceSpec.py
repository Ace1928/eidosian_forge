from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetReleaseResourceSpec():
    """Constructs and returns the Resource specification for Delivery Pipeline."""
    return concepts.ResourceSpec('clouddeploy.projects.locations.deliveryPipelines.releases', resource_name='release', deliveryPipelinesId=DeliveryPipelineAttributeConfig(), releasesId=ReleaseAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=LocationAttributeConfig(), disable_auto_completers=False)