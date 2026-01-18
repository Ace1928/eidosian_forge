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
def AddNodeGroupResourceArg(parser, verb, api_version):
    """Adds node group resource argument to parser."""

    def NodeGroupConfig():
        return concepts.ResourceParameterAttributeConfig(name='node_group', help_text='Node group ID.')

    def GetNodeGroupResourceSpec(api_version):
        return concepts.ResourceSpec('dataproc.projects.regions.clusters.nodeGroups', api_version=api_version, resource_name='node_group', disable_auto_completers=True, projectId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, region=_RegionAttributeConfig(), clusterName=ClusterConfig(), nodeGroupsId=NodeGroupConfig())
    concept_parsers.ConceptParser.ForResource('node_group', GetNodeGroupResourceSpec(api_version), 'ID of the node group to {0}.'.format(verb), required=True).AddToParser(parser)