from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
def GetShareWithFlag(custom_name=None):
    """Gets the --share-with flag."""
    help_text = '  If this future reservation is shared, provide a comma-separated list\n  of projects that this future reservation is shared with.\n  The list must contain project IDs or project numbers.\n  '
    return base.Argument(custom_name if custom_name else '--share-with', type=arg_parsers.ArgList(min_length=1), metavar='PROJECT', help=help_text)