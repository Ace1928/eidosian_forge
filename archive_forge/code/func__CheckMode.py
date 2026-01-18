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
def _CheckMode(name, mode_value, mount_disk, matching_disk, create):
    """Make sure the correct mode is specified for container mount disk."""
    partition = mount_disk.get('partition')
    if mode_value == containers_utils.MountVolumeMode.READ_WRITE and matching_disk.get('ro'):
        raise exceptions.InvalidArgumentException('--container-mount-disk', 'Value for [mode] in [--container-mount-disk] cannot be [rw] if the disk is attached in [ro] mode: disk name [{}], partition [{}]'.format(name, partition))
    if matching_disk.get('ro') and create:
        raise exceptions.InvalidArgumentException('--container-mount-disk', 'Cannot mount disk named [{}] to container: disk is created in [ro] mode and thus cannot be formatted.'.format(name))