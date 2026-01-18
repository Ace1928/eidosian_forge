from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
def GetStartTimeFlag(required=True):
    """Gets the --start-time flag."""
    return base.Argument('--start-time', required=required, type=str, help=GetStartTimeHelpText())