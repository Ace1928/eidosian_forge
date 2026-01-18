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
def MergeContacts(api_version, prev_contacts, new_contacts):
    domains_messages = registrations.GetMessagesModule(api_version)
    if new_contacts is None:
        new_contacts = domains_messages.ContactSettings()
    return domains_messages.ContactSettings(registrantContact=new_contacts.registrantContact or prev_contacts.registrantContact, adminContact=new_contacts.adminContact or prev_contacts.adminContact, technicalContact=new_contacts.technicalContact or prev_contacts.technicalContact)