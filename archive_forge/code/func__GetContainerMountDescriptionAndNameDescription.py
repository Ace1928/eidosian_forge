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
def _GetContainerMountDescriptionAndNameDescription(for_update=False):
    """Get description text for --container-mount-disk flag."""
    if for_update:
        description = 'Mounts a disk to the container by using mount-path or updates how the volume is\nmounted if the same mount path has already been declared. The disk must already\nbe attached to the instance with a device-name that matches the disk name.\nMultiple flags are allowed.\n'
        name_description = 'Name of the disk. Can be omitted if exactly one additional disk is attached to\nthe instance. The name of the single additional disk will be used by default.\n'
        return (description, name_description)
    else:
        description = "Mounts a disk to the specified mount path in the container. Multiple '\nflags are allowed. Must be used with `--disk` or `--create-disk`.\n"
        name_description = 'Name of the disk. If exactly one additional disk is attached\nto the instance using `--disk` or `--create-disk`, specifying disk\nname here is optional. The name of the single additional disk will be\nused by default.\n'
        return (description, name_description)