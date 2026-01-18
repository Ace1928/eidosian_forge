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
def ValidateBulkCreateArgs(args):
    """Validates args for bulk create."""
    if args.IsSpecified('name_pattern') and (not args.IsSpecified('count')):
        raise exceptions.RequiredArgumentException('--count', 'The `--count` argument must be specified when the `--name-pattern` argument is specified.')
    if args.IsSpecified('location_policy') and (args.IsSpecified('zone') or not args.IsSpecified('region')):
        raise exceptions.RequiredArgumentException('--region', 'The `--region` argument must be used alongside the `--location-policy` argument and not `--zone`.')