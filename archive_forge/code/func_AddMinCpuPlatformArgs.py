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
def AddMinCpuPlatformArgs(parser, track, required=False):
    parser.add_argument('--min-cpu-platform', metavar='PLATFORM', required=required, help='      When specified, the VM will be scheduled on host with specified CPU\n      architecture or a newer one. To list available CPU platforms in given\n      zone, run:\n\n          $ gcloud {}compute zones describe ZONE --format="value(availableCpuPlatforms)"\n\n      Default setting is "AUTOMATIC".\n\n      CPU platform selection is available only in selected zones.\n\n      You can find more information on-line:\n      [](https://cloud.google.com/compute/docs/instances/specify-min-cpu-platform)\n      '.format(track.prefix + ' ' if track.prefix else ''))