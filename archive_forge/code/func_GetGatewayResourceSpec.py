from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetGatewayResourceSpec(resource_name='gateway'):
    return concepts.ResourceSpec('apigateway.projects.locations.gateways', resource_name=resource_name, gatewaysId=GatewayAttributeConfig(), locationsId=LocationAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)