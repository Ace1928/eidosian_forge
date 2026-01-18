from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from apitools.base.protorpclite import messages as _messages
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.command_lib.domains import flags
from googlecloudsdk.command_lib.domains import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
def PromptForPublicContactsAck(domain, contacts, print_format='default'):
    """Asks a user for Public Contacts Ack.

  Args:
    domain: Domain name.
    contacts: Current Contacts. All 3 contacts should be present.
    print_format: Print format, e.g. 'default' or 'yaml'.

  Returns:
    Boolean: whether the user accepted the notice or not.
  """
    log.status.Print("You choose to make contact data of domain {} public.\nAnyone who looks it up in the WHOIS directory will be able to see info\nfor the domain owner and administrative and technical contacts.\nMake sure it's ok with them that their contact data is public.\nThis info will be publicly available:".format(domain))
    contacts = _SimplifyContacts(contacts)
    resource_printer.Print(contacts, print_format, out=sys.stderr)
    return console_io.PromptContinue(message=None, default=False, throw_if_unattended=True, cancel_on_no=True)