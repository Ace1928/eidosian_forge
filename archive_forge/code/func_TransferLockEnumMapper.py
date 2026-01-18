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
def TransferLockEnumMapper(domains_messages):
    return arg_utils.ChoiceEnumMapper('--transfer-lock-state', _GetTransferLockEnum(domains_messages), custom_mappings={'LOCKED': ('locked', 'The transfer lock is locked.'), 'UNLOCKED': ('unlocked', 'The transfer lock is unlocked.')}, required=False, help_str='Transfer Lock of a registration. It needs to be unlocked in order to transfer the domain to another registrar.')