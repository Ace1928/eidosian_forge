from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.disks import flags as disks_flags
from googlecloudsdk.command_lib.util import completers
def MakeForceArg():
    return base.Argument('--force', action='store_true', default=False, help='          By default, image creation fails when it is created from a disk that\n          is attached to a running instance. When this flag is used, image\n          creation from disk will proceed even if the disk is in use.\n          ')