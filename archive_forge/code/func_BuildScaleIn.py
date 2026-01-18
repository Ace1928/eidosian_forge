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
def BuildScaleIn(args, messages):
    """Builds AutoscalingPolicyScaleInControl.

  Args:
    args: command line arguments.
    messages: module containing message classes.

  Returns:
    AutoscalingPolicyScaleInControl message object.
  Raises:
    InvalidArgumentError:  if both max-scaled-in-replicas and
      max-scaled-in-replicas-percent are specified.
  """
    if args.IsSpecified('scale_in_control'):
        replicas_arg = args.scale_in_control.get('max-scaled-in-replicas')
        replicas_arg_percent = args.scale_in_control.get('max-scaled-in-replicas-percent')
        if replicas_arg and replicas_arg_percent:
            raise InvalidArgumentError("max-scaled-in-replicas and max-scaled-in-replicas-percentare mutually exclusive, you can't specify both")
        elif replicas_arg_percent:
            max_replicas = messages.FixedOrPercent(percent=int(replicas_arg_percent))
        else:
            max_replicas = messages.FixedOrPercent(fixed=int(replicas_arg))
        return messages.AutoscalingPolicyScaleInControl(maxScaledInReplicas=max_replicas, timeWindowSec=args.scale_in_control.get('time-window'))