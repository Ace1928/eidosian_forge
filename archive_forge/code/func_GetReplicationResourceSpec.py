from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def GetReplicationResourceSpec():
    location_attribute_config = GetLocationAttributeConfig()
    volume_attribute_config = GetVolumeAttributeConfig(positional=False)
    return concepts.ResourceSpec(constants.REPLICATIONS_COLLECTION, resource_name='replication', api_version=constants.BETA_API_VERSION, projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=location_attribute_config, volumesId=volume_attribute_config, replicationsId=GetReplicationAttributeConfig())