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
def AddAutoscalingPolicyResourceArgForCluster(parser, api_version):
    """Adds a workflow template resource argument.

  Args:
    parser: the argparse parser for the command.
    api_version: api version, for example v1
  """
    concept_parsers.ConceptParser.ForResource('--autoscaling-policy', _AutoscalingPolicyResourceSpec(api_version), 'The autoscaling policy to use.', command_level_fallthroughs={'region': ['--region']}, flag_name_overrides={'region': ''}, required=False).AddToParser(parser)