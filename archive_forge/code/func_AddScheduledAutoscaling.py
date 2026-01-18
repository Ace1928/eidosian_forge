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
def AddScheduledAutoscaling(parser, patch_args):
    """Add parameters controlling scheduled autoscaling."""
    if patch_args:
        arg_group = parser.add_group(mutex=True)
        arg_group_config = parser.add_group()
        arg_group.add_argument('--set-schedule', metavar='SCHEDULE_NAME', help='A unique name for the scaling schedule to be configured.')
        arg_group.add_argument('--update-schedule', metavar='SCHEDULE_NAME', help='Name of the scaling schedule to be updated.')
        arg_group.add_argument('--remove-schedule', metavar='SCHEDULE_NAME', help="          Name of the scaling schedule to be removed.\n\n          Be careful with this action as scaling schedule deletion cannot be\n          undone.\n\n          You can delete any schedule regardless of its status. If you delete\n          a scaling schedule that is currently active, the deleted scaling\n          schedule stops being effective immediately after it is deleted.\n          If there is no need to maintain capacity, the autoscaler starts\n          removing instances after the usual stabilization period and after\n          scale-in controls (if configured). For more information, see\n          [Delays in scaling in](https://cloud.google.com/compute/docs/autoscaler/understanding-autoscaler-decisions#delays_in_scaling_in) and [Scale-in controls](https://cloud.google.com/compute/docs/autoscaler/understanding-autoscaler-decisions#scale-in_controls).\n          This ensures you don't accidentally lose capacity immediately after\n          the scaling schedule ends.\n          ")
        arg_group.add_argument('--enable-schedule', metavar='SCHEDULE_NAME', help='        Name of the scaling schedule to be enabled.\n\n        See --disable-schedule for details.\n        ')
        arg_group.add_argument('--disable-schedule', metavar='SCHEDULE_NAME', help="          Name of the scaling schedule to be disabled.\n\n          When a scaling schedule is disabled its configuration persists but\n          the scaling schedule itself never becomes active. If you disable a\n          scaling schedule that is currently active the disabled scaling\n          schedule stops being effective immediately after it moves into\n          DISABLED state.\n          If there is no need to maintain capacity, the autoscaler starts\n          removing instances after the usual stabilization period and after\n          scale-in controls (if configured). For more information, see\n          [Delays in scaling in](https://cloud.google.com/compute/docs/autoscaler/understanding-autoscaler-decisions#delays_in_scaling_in) and [Scale-in controls](https://cloud.google.com/compute/docs/autoscaler/understanding-autoscaler-decisions#scale-in_controls).\n          This ensures you don't accidentally lose capacity immediately after\n          the scaling schedule ends.\n          ")
        AddScheduledAutoscalingConfigurationArguments(arg_group_config)
    else:
        arg_group = parser
        parser.add_argument('--set-schedule', metavar='SCHEDULE_NAME', help='Unique name for the scaling schedule.')
        AddScheduledAutoscalingConfigurationArguments(parser.add_group())