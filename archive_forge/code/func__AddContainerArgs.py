from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import functools
import ipaddress
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
import six
def _AddContainerArgs(parser):
    """Add basic args for update-container."""
    parser.add_argument('--container-image', type=NonEmptyString('--container-image'), help='      Sets container image in the declaration to the specified value.\n\n      Empty string is not allowed.\n      ')
    parser.add_argument('--container-privileged', action='store_true', default=None, help='      Sets permission to run container to the specified value.\n      ')
    parser.add_argument('--container-stdin', action='store_true', default=None, help='      Sets configuration whether to keep container `STDIN` always open to the\n      specified value.\n      ')
    parser.add_argument('--container-tty', action='store_true', default=None, help='      Sets configuration whether to allocate a pseudo-TTY for the container\n      to the specified value.\n      ')
    parser.add_argument('--container-restart-policy', choices=['never', 'on-failure', 'always'], metavar='POLICY', type=lambda val: val.lower(), help='      Sets container restart policy to the specified value.\n      ')