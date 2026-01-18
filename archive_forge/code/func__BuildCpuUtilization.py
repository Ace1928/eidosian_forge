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
def _BuildCpuUtilization(args, messages):
    """Builds the CPU Utilization message given relevant arguments."""
    flags_to_check = ['target_cpu_utilization', 'scale_based_on_cpu', 'cpu_utilization_predictive_method']
    if instance_utils.IsAnySpecified(args, *flags_to_check):
        cpu_message = messages.AutoscalingPolicyCpuUtilization()
        if args.target_cpu_utilization:
            cpu_message.utilizationTarget = args.target_cpu_utilization
        if args.cpu_utilization_predictive_method:
            cpu_predictive_enum = messages.AutoscalingPolicyCpuUtilization.PredictiveMethodValueValuesEnum
            cpu_message.predictiveMethod = arg_utils.ChoiceToEnum(args.cpu_utilization_predictive_method, cpu_predictive_enum)
        return cpu_message
    return None