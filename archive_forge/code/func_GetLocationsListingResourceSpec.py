from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetLocationsListingResourceSpec():
    """Gets the location resource spec for listing resources."""
    fallthroughs = [deps.ArgFallthrough('--location'), deps.PropertyFallthrough(properties.VALUES.edge_container.location)]
    config = LocationAttributeConfig()
    config.fallthroughs = fallthroughs
    return concepts.ResourceSpec('edgecontainer.projects.locations', resource_name='location', locationsId=config, projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)