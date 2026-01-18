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
def ParseContactPrivacy(api_version, contact_privacy):
    domains_messages = registrations.GetMessagesModule(api_version)
    if contact_privacy is None:
        return None
    return flags.ContactPrivacyEnumMapper(domains_messages).GetEnumForChoice(contact_privacy)