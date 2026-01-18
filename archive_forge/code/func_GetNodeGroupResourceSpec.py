from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import json
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
import six
def GetNodeGroupResourceSpec(api_version):
    return concepts.ResourceSpec('dataproc.projects.regions.clusters.nodeGroups', api_version=api_version, resource_name='node_group', disable_auto_completers=True, projectId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, region=_RegionAttributeConfig(), clusterName=ClusterConfig(), nodeGroupsId=NodeGroupConfig())