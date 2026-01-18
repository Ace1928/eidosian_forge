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
def AddAutoscalerArgs(parser, autoscaling_file_enabled=False, patch_args=False):
    """Adds commandline arguments to parser."""
    parser.add_argument('--cool-down-period', type=arg_parsers.Duration(), help="The number of seconds that your application takes to initialize on a VM instance. This is referred to as the [initialization period](https://cloud.google.com/compute/docs/autoscaler#cool_down_period). Specifying an accurate initialization period improves autoscaler decisions. For example, when scaling out, the autoscaler ignores data from VMs that are still initializing because those VMs might not yet represent normal usage of your application. The default initialization period is 60 seconds. See $ gcloud topic datetimes for information on duration formats. Initialization periods might vary because of numerous factors. We recommend that you test how long your application may take to initialize. To do this, create a VM and time your application's startup process.")
    parser.add_argument('--description', help='Notes about Autoscaler.')
    AddMinMaxControl(parser, max_required=not autoscaling_file_enabled)
    parser.add_argument('--scale-based-on-cpu', action='store_true', help='Autoscaler will be based on CPU utilization.')
    parser.add_argument('--scale-based-on-load-balancing', action='store_true', help='Use autoscaling based on load balancing utilization.')
    parser.add_argument('--target-cpu-utilization', type=arg_parsers.BoundedFloat(0.0, 1.0), help='Autoscaler aims to maintain CPU utilization at target level (0.0 to 1.0).')
    parser.add_argument('--target-load-balancing-utilization', type=arg_parsers.BoundedFloat(0.0, None), help='Autoscaler aims to maintain the load balancing utilization level (greater than 0.0).')
    custom_metric_utilization_help = 'Adds a target metric value for the Autoscaler to use.\n\n*metric*::: Protocol-free URL of a Google Cloud Monitoring metric.\n\n*utilization-target*::: Value of the metric Autoscaler aims to\n  maintain (greater than 0.0).\n\n*utilization-target-type*::: How target is expressed. Valid values: {0}.\n\nMutually exclusive with `--update-stackdriver-metric`.\n'.format(', '.join(_ALLOWED_UTILIZATION_TARGET_TYPES))
    parser.add_argument('--custom-metric-utilization', type=arg_parsers.ArgDict(spec={'metric': str, 'utilization-target': float, 'utilization-target-type': str}), action='append', help=custom_metric_utilization_help)
    if autoscaling_file_enabled:
        parser.add_argument('--autoscaling-file', metavar='PATH', help='Path of the file from which autoscaling configuration will be loaded. This flag allows you to atomically setup complex autoscalers.')
    parser.add_argument('--remove-stackdriver-metric', metavar='METRIC', help='Stackdriver metric to remove from autoscaling configuration. If the metric is the only input used for autoscaling the command will fail.')
    parser.add_argument('--update-stackdriver-metric', metavar='METRIC', help='Stackdriver metric to use as an input for autoscaling. When using this flag, the target value of the metric must also be specified by using the following flags: `--stackdriver-metric-single-instance-assignment` or `--stackdriver-metric-utilization-target` and `--stackdriver-metric-utilization-target-type`. Mutually exclusive with `--custom-metric-utilization`.')
    parser.add_argument('--stackdriver-metric-filter', metavar='FILTER', help='Expression for filtering samples used to autoscale, see https://cloud.google.com/monitoring/api/v3/filters.')
    parser.add_argument('--stackdriver-metric-utilization-target', metavar='TARGET', type=float, help='Value of the metric Autoscaler aims to maintain. When specifying this flag you must also provide `--stackdriver-metric-utilization-target-type`. Mutually exclusive with `--stackdriver-metric-single-instance-assignment` and `--custom-metric-utilization`.')
    parser.add_argument('--stackdriver-metric-utilization-target-type', metavar='TARGET_TYPE', choices=_ALLOWED_UTILIZATION_TARGET_TYPES_LOWER, help='Value of the metric Autoscaler aims to maintain. When specifying this flag you must also provide `--stackdriver-metric-utilization-target`. Mutually exclusive with `--stackdriver-metric-single-instance-assignment` and `--custom-metric-utilization`.')
    parser.add_argument('--stackdriver-metric-single-instance-assignment', metavar='ASSIGNMENT', type=float, help='Value that indicates the amount of work that each instance is expected to handle. Autoscaler maintains enough VMs by dividing the available work by this value. Mutually exclusive with `-stackdriver-metric-utilization-target-type`, `-stackdriver-metric-utilization-target-type`, and `--custom-metric-utilization`.')
    GetModeFlag().AddToParser(parser)
    AddScaleInControlFlag(parser)
    AddScheduledAutoscaling(parser, patch_args)