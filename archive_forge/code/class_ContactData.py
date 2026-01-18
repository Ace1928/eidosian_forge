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
class ContactData(_messages.Message):
    """Message that should be present in YAML file with contacts data."""
    allContacts = _messages.MessageField(domains_messages.Contact, 1)
    registrantContact = _messages.MessageField(domains_messages.Contact, 2)
    adminContact = _messages.MessageField(domains_messages.Contact, 3)
    technicalContact = _messages.MessageField(domains_messages.Contact, 4)