from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetAdminEnabledFlag(args):
    """Determines value of admin_enabled/enable_admin flag.

  Args:
    args: A list of arguments to be parsed.

  Returns:
    A boolean indicates whether admin mode is enabled in Arguments.
  """
    return args.enable_admin if args.enable_admin is not None else args.admin_enabled