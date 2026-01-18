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
def _UpdateCustomMetricUtilizationsFromStackoverflowFlags(args, messages, original):
    """Take apply stackdriver flags to customMetricUtilizations."""
    if original:
        result = original.autoscalingPolicy.customMetricUtilizations
    else:
        result = []
    if args.remove_stackdriver_metric:
        _RemoveMetricFromList(result, args.remove_stackdriver_metric)
    if args.update_stackdriver_metric:
        _RemoveMetricFromList(result, args.update_stackdriver_metric)
        if args.stackdriver_metric_utilization_target_type:
            target_type = messages.AutoscalingPolicyCustomMetricUtilization.UtilizationTargetTypeValueValuesEnum(args.stackdriver_metric_utilization_target_type.upper().replace('-', '_'))
        else:
            target_type = None
        if args.stackdriver_metric_filter and "'" in args.stackdriver_metric_filter:
            log.warning('The provided filter contains a single quote character (\'). While valid as a metric/resource label value, it\'s not a control character that is part of the filtering language; if you meant to use it to quote a string value, you need to use a double quote character (") instead.')
        result.append(messages.AutoscalingPolicyCustomMetricUtilization(utilizationTarget=args.stackdriver_metric_utilization_target, metric=args.update_stackdriver_metric, utilizationTargetType=target_type, singleInstanceAssignment=args.stackdriver_metric_single_instance_assignment, filter=args.stackdriver_metric_filter))
    return result