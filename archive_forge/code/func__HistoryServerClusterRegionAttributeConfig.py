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
def _HistoryServerClusterRegionAttributeConfig():
    return concepts.ResourceParameterAttributeConfig(name='history-server-cluster-region', help_text='Compute Engine region for the {resource}. It must be the same region as the Dataproc cluster that is being created.', fallthroughs=_DataprocRegionFallthrough())