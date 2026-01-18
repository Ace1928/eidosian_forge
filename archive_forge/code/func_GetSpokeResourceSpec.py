from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.network_connectivity import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetSpokeResourceSpec(location_arguments):
    return concepts.ResourceSpec('networkconnectivity.projects.locations.spokes', resource_name='spoke', spokesId=SpokeAttributeConfig(), locationsId=LocationAttributeConfig(location_arguments), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)