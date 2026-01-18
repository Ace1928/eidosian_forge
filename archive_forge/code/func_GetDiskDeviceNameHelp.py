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
def GetDiskDeviceNameHelp(container_mount_enabled=False):
    """Helper to get documentation for "device-name" param of disk spec."""
    if container_mount_enabled:
        return 'An optional name to display the disk name in the guest operating system. Must be the same as `name` if used with `--container-mount-disk`. If omitted, a device name of the form `persistent-disk-N` is used. If omitted and used with `--container-mount-disk` (where the `name` of the container mount disk is the same as in this flag), a device name equal to disk `name` is used.'
    else:
        return 'An optional name to display the disk name in the guest operating system. If omitted, a device name of the form `persistent-disk-N` is used.'