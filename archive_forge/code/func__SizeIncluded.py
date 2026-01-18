from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import disks_util as api_util
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.disks import flags as disks_flags
from googlecloudsdk.command_lib.util.args import labels_util
def _SizeIncluded(args):
    return args.IsSpecified('size')