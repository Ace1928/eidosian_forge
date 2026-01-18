from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import functools
import itertools
import sys
import textwrap
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.api_lib.ml_engine import versions_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.iam import completers as iam_completers
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.ml_engine import constants
from googlecloudsdk.command_lib.ml_engine import models_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def AddAutoScalingFlags(parser):
    """Adds flags related to autoscaling to the specified parser."""
    autoscaling_group = parser.add_argument_group(help='Configure the autoscaling settings to be deployed.')
    autoscaling_group.add_argument('--min-nodes', type=int, help='The minimum number of nodes to scale this model under load.\n')
    autoscaling_group.add_argument('--max-nodes', type=int, help='The maximum number of nodes to scale this model under load.\n')
    autoscaling_group.add_argument('--metric-targets', metavar='METRIC-NAME=TARGET', type=arg_parsers.ArgDict(key_type=_ValidateMetricTargetKey, value_type=_ValidateMetricTargetValue), action=arg_parsers.UpdateAction, default={}, help="List of key-value pairs to set as metrics' target for autoscaling.\nAutoscaling could be based on CPU usage or GPU duty cycle, valid key could be\ncpu-usage or gpu-duty-cycle.\n")