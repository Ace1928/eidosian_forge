from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.deploy import automation_util
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _CreateDeliveryPipelineResource(messages, delivery_pipeline_name, project, region):
    """Creates delivery pipeline resource with full delivery pipeline name and the resource reference."""
    resource = messages.DeliveryPipeline()
    resource_ref = resources.REGISTRY.Parse(delivery_pipeline_name, collection='clouddeploy.projects.locations.deliveryPipelines', params={'projectsId': project, 'locationsId': region, 'deliveryPipelinesId': delivery_pipeline_name})
    resource.name = resource_ref.RelativeName()
    return (resource, resource_ref)