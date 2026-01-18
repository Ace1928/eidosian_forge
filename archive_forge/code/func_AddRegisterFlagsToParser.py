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
def AddRegisterFlagsToParser(parser):
    """Get flags for registering a domain.

  Args:
    parser: argparse parser to which to add these flags.
  """
    _AddDNSSettingsFlagsToParser(parser, mutation_op=MutationOp.REGISTER)
    _AddContactSettingsFlagsToParser(parser, mutation_op=MutationOp.REGISTER)
    _AddPriceFlagsToParser(parser, MutationOp.REGISTER)
    messages = apis.GetMessagesModule('domains', API_VERSION_FOR_FLAGS)
    notice_choices = ContactNoticeEnumMapper(messages).choices.copy()
    notice_choices.update({'hsts-preloaded': 'By sending this notice you acknowledge that the domain is preloaded on the HTTP Strict Transport Security list in browsers. Serving a website on such domain will require an SSL certificate. See https://support.google.com/domains/answer/7638036 for details.'})
    base.Argument('--notices', help='Notices about special properties of certain domains or contacts.', metavar='NOTICE', type=arg_parsers.ArgList(element_type=str, choices=notice_choices)).AddToParser(parser)