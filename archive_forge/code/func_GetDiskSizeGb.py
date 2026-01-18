from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import textwrap
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.api_lib.compute.regions import utils as region_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.disks import create
from googlecloudsdk.command_lib.compute.disks import flags as disks_flags
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.compute.resource_policies import flags as resource_flags
from googlecloudsdk.command_lib.compute.resource_policies import util as resource_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
import six
def GetDiskSizeGb(self, args, from_image):
    size_gb = utils.BytesToGb(args.size)
    if size_gb:
        if args.type in constants.LEGACY_DISK_TYPE_LIST and size_gb < 10:
            raise exceptions.InvalidArgumentException('--size', 'Value must be greater than or equal to 10 GB; reveived {0} GB'.format(size_gb))
        pass
    elif args.source_snapshot or from_image or args.source_disk or self.GetFromSourceInstantSnapshot(args):
        pass
    elif args.type in constants.DEFAULT_DISK_SIZE_GB_MAP:
        size_gb = constants.DEFAULT_DISK_SIZE_GB_MAP[args.type]
    elif args.type:
        pass
    else:
        size_gb = constants.DEFAULT_DISK_SIZE_GB_MAP[constants.DISK_TYPE_PD_STANDARD]
    utils.WarnIfDiskSizeIsTooSmall(size_gb, args.type)
    return size_gb