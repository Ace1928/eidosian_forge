from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import random
import re
import string
import sys
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import auto_healing_utils
from googlecloudsdk.command_lib.compute.managed_instance_groups import update_instances_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _BuildAutoscalerPolicy(args, messages, original):
    """Builds AutoscalingPolicy from args.

  Args:
    args: command line arguments.
    messages: module containing message classes.
    original: original autoscaler message.

  Returns:
    AutoscalingPolicy message object.
  """
    policy_dict = {'coolDownPeriodSec': args.cool_down_period, 'cpuUtilization': _BuildCpuUtilization(args, messages), 'customMetricUtilizations': _BuildCustomMetricUtilizations(args, messages, original), 'loadBalancingUtilization': _BuildLoadBalancingUtilization(args, messages), 'maxNumReplicas': args.max_num_replicas, 'minNumReplicas': args.min_num_replicas}
    policy_dict['mode'] = _BuildMode(args, messages, original)
    policy_dict['scaleInControl'] = BuildScaleIn(args, messages)
    policy_dict['scalingSchedules'] = BuildSchedules(args, messages)
    return messages.AutoscalingPolicy(**dict(((key, value) for key, value in six.iteritems(policy_dict) if value is not None)))