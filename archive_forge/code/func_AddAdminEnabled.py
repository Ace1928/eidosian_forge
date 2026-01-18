from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddAdminEnabled(parser, default_behavior=True, update=False):
    """Adds adminEnabled flag to the argparse.ArgumentParser.

  Args:
    parser: The argparse parser.
    default_behavior: A boolean indicates whether command allows user to set
      Administrative status.
    update: A boolean indicates whether the incoming request is update.
  """
    group = parser.add_group(mutex=True, required=False, help='')
    if update:
        help_text = '      Administrative status of the interconnect attachment.\n      When this is enabled, the attachment is operational and will carry\n      traffic. Use --no-enable-admin to disable it.\n      '
    elif default_behavior:
        help_text = '      Administrative status of the interconnect attachment. If not provided\n      on creation, defaults to enabled.\n      When this is enabled, the attachment is operational and will carry\n      traffic. Use --no-enable-admin to disable it.\n      '
    else:
        help_text = '      Administrative status of the interconnect attachment. If not provided\n      on creation, defaults to disabled.\n      When this is enabled, the attachment is operational and will carry\n      traffic. Use --no-enable-admin to disable it.\n      '
    group.add_argument('--admin-enabled', hidden=True, default=None, action='store_true', help='(DEPRECATED) Use --enable-admin instead.')
    group.add_argument('--enable-admin', action='store_true', default=None, help=help_text)