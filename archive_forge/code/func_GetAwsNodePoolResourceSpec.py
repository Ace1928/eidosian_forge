from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetAwsNodePoolResourceSpec():
    return concepts.ResourceSpec('gkemulticloud.projects.locations.awsClusters.awsNodePools', resource_name='node_pool', awsNodePoolsId=AwsNodePoolAttributeConfig(), awsClustersId=AwsClusterAttributeConfig(), locationsId=LocationAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)