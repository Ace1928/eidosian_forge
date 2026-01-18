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
def _AddContainerMountHostPathFlag(parser, for_update=False):
    """Helper to add --container-mount-host-path flag."""
    if for_update:
        additional = '\n      - Adds a volume, if `mount-path` is not yet declared.\n      - Replaces a volume, if `mount-path` is declared.\n      All parameters (`host-path`, `mount-path`, `mode`) are completely\n      replaced.'
    else:
        additional = ''
    parser.add_argument('--container-mount-host-path', metavar='host-path=HOSTPATH,mount-path=MOUNTPATH[,mode=MODE]', type=arg_parsers.ArgDict(spec={'host-path': str, 'mount-path': str, 'mode': functools.partial(ParseMountVolumeMode, '--container-mount-host-path')}), action='append', help='      Mounts a volume by using host-path.{}\n\n      *host-path*::: Path on host to mount from.\n\n      *mount-path*::: Path on container to mount to. Mount paths with spaces\n      and commas (and other special characters) are not supported by this\n      command.\n\n      *mode*::: Volume mount mode: rw (read/write) or ro (read-only).\n\n      Default: rw.\n      '.format(additional))