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
def _ValidateAndParseDiskRefsRegionalReplica(args, compute_holder):
    """Validate flags and parse disks references.

  Subclasses may override it to customize parsing.

  Args:
    args: The argument namespace
    compute_holder: base_classes.ComputeApiHolder instance

  Returns:
    List of compute.regionDisks resources.
  """
    if not args.IsSpecified('replica_zones') and args.IsSpecified('region') and (not args.IsSpecified('source_instant_snapshot')):
        raise exceptions.RequiredArgumentException('--replica-zones', '--replica-zones is required for regional disk creation')
    if args.replica_zones is not None:
        return create.ParseRegionDisksResources(compute_holder.resources, args.DISK_NAME, args.replica_zones, args.project, args.region)
    disk_refs = Create.disks_arg.ResolveAsResource(args, compute_holder.resources, scope_lister=flags.GetDefaultScopeLister(compute_holder.client))
    for disk_ref in disk_refs:
        if disk_ref.Collection() == 'compute.regionDisks' and (not args.IsSpecified('source_instant_snapshot')):
            raise exceptions.RequiredArgumentException('--replica-zones', '--replica-zones is required for regional disk creation [{}]'.format(disk_ref.SelfLink()))
    return disk_refs