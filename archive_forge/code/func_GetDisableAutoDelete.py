from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
def GetDisableAutoDelete():
    """Gets the --disable-auto-delete flag."""
    help_text = '  Disables the auto-delete setting for the reservation.\n  '
    return base.Argument('--disable-auto-delete', action='store_true', help=help_text)