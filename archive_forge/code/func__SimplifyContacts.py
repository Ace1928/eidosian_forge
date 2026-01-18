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
def _SimplifyContacts(contacts):
    """Returns one contact if all 3 contacts are equal, and all 3 contacts otherwise."""
    if contacts.registrantContact == contacts.adminContact and contacts.registrantContact == contacts.technicalContact:
        return contacts.registrantContact
    return contacts