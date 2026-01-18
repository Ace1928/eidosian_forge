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
def AddContainerMountDiskFlag(parser, for_update=False):
    """Add --container-mount-disk flag."""
    description, name_description = _GetContainerMountDescriptionAndNameDescription(for_update=for_update)
    help_text = '{}\n\n*name*::: {}\n\n*mount-path*::: Path on container to mount to. Mount paths with spaces\n      and commas (and other special characters) are not supported by this\n      command.\n\n*partition*::: Optional. The partition of the disk to mount. Multiple\npartitions of a disk can be mounted.{}\n\n*mode*::: Volume mount mode: `rw` (read/write) or `ro` (read-only).\nDefaults to `rw`. Fails if the disk mode is `ro` and volume mount mode\nis `rw`.\n'.format(description, name_description, '' if for_update else " Can't be used with --create-disk.")
    spec = {'name': str, 'mount-path': str, 'partition': int, 'mode': functools.partial(ParseMountVolumeMode, '--container-mount-disk')}
    parser.add_argument('--container-mount-disk', type=arg_parsers.ArgDict(spec=spec, required_keys=['mount-path']), help=help_text, action='append')