from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags as maintenance_flags
from googlecloudsdk.command_lib.util.args import labels_util
def ValidateBulkTargetShapeArgs(args):
    """Validates target shape arg for bulk create."""
    if args.IsSpecified('target_distribution_shape') and (args.IsSpecified('zone') or not args.IsSpecified('region')):
        raise exceptions.RequiredArgumentException('--region', 'The `--region` argument must be used alongside the `--target_distribution_shape` argument and not `--zone`.')