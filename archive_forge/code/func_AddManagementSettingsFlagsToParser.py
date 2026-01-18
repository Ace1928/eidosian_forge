from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddManagementSettingsFlagsToParser(parser):
    """Get flags for configure management command.

  Args:
    parser: argparse parser to which to add these flags.
  """
    messages = apis.GetMessagesModule('domains', API_VERSION_FOR_FLAGS)
    TransferLockEnumMapper(messages).choice_arg.AddToParser(parser)
    RenewalMethodEnumMapper(messages).choice_arg.AddToParser(parser)