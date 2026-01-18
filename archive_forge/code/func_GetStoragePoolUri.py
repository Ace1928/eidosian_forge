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
def GetStoragePoolUri(self, args, compute_holder):
    if args.storage_pool:
        storage_pool_ref = disks_flags.STORAGE_POOL_ARG.ResolveAsResource(args, compute_holder.resources, default_scope=compute_scope.ScopeEnum.ZONE)
        if storage_pool_ref:
            return storage_pool_ref.SelfLink()
    return None