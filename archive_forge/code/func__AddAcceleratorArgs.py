from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.dataproc import compute_helpers
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from stdin.
def _AddAcceleratorArgs(parser, include_driver_pool_args=False):
    """Adds accelerator related args to the parser."""
    accelerator_help_fmt = '      Attaches accelerators, such as GPUs, to the {instance_type}\n      instance(s).\n      '
    accelerator_help_fmt += '\n      *type*::: The specific type of accelerator to attach to the instances,\n      such as `nvidia-tesla-k80` for NVIDIA Tesla K80. Use `gcloud compute\n      accelerator-types list` to display available accelerator types.\n\n      *count*::: The number of accelerators to attach to each instance. The default value is 1.\n      '
    parser.add_argument('--master-accelerator', type=arg_parsers.ArgDict(spec={'type': str, 'count': int}), metavar='type=TYPE,[count=COUNT]', help=accelerator_help_fmt.format(instance_type='master'))
    parser.add_argument('--worker-accelerator', type=arg_parsers.ArgDict(spec={'type': str, 'count': int}), metavar='type=TYPE,[count=COUNT]', help=accelerator_help_fmt.format(instance_type='worker'))
    secondary_worker_accelerator = parser.add_argument_group(mutex=True)
    secondary_worker_accelerator.add_argument('--secondary-worker-accelerator', type=arg_parsers.ArgDict(spec={'type': str, 'count': int}), metavar='type=TYPE,[count=COUNT]', help=accelerator_help_fmt.format(instance_type='secondary-worker'))
    secondary_worker_accelerator.add_argument('--preemptible-worker-accelerator', type=arg_parsers.ArgDict(spec={'type': str, 'count': int}), metavar='type=TYPE,[count=COUNT]', help=accelerator_help_fmt.format(instance_type='preemptible-worker'), hidden=True, action=actions.DeprecationAction('--preemptible-worker-accelerator', warn='The `--preemptible-worker-accelerator` flag is deprecated. Use the `--secondary-worker-accelerator` flag instead.'))
    if include_driver_pool_args:
        parser.add_argument('--driver-pool-accelerator', type=arg_parsers.ArgDict(spec={'type': str, 'count': int}), metavar='type=TYPE,[count=COUNT]', help=accelerator_help_fmt.format(instance_type='driver-pool'))