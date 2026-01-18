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
def HistoryServerClusterConfig():
    return concepts.ResourceParameterAttributeConfig(name='history-server-cluster', help_text='Spark History Server. Resource name of an existing Dataproc cluster to act as a Spark History Server for workloads run on the Cluster.')