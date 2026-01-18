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
def _PromptForSingleContact(domains_messages, unused_current_contact=None):
    """Asks a user for a single contact data."""
    contact = domains_messages.Contact()
    contact.postalAddress = domains_messages.PostalAddress()
    contact.postalAddress.recipients.append(util.PromptWithValidator(validator=util.ValidateNonEmpty, error_message=' Name must not be empty.', prompt_string='Full name:  '))
    contact.postalAddress.organization = console_io.PromptResponse('Organization (if applicable):  ')
    contact.email = util.PromptWithValidator(validator=util.ValidateEmail, error_message=' Invalid email address.', prompt_string='Email', default=properties.VALUES.core.account.Get())
    contact.phoneNumber = util.PromptWithValidator(validator=util.ValidateNonEmpty, error_message=' Phone number must not be empty.', prompt_string='Phone number:  ', message='Enter phone number with country code, e.g. "+1.8005550123".')
    contact.faxNumber = util.Prompt(prompt_string='Fax number (if applicable):  ', message='Enter fax number with country code, e.g. "+1.8005550123".')
    contact.postalAddress.regionCode = util.PromptWithValidator(validator=util.ValidateRegionCode, error_message=' Country / Region code must be in ISO 3166-1 format, e.g. "US" or "PL".\n See https://support.google.com/business/answer/6270107 for a list of valid choices.', prompt_string='Country / Region code:  ', message='Enter two-letter Country / Region code, e.g. "US" or "PL".')
    if contact.postalAddress.regionCode != 'US':
        log.status.Print('Refer to the guidelines for entering address field information at https://support.google.com/business/answer/6397478.')
    contact.postalAddress.postalCode = console_io.PromptResponse('Postal / ZIP code:  ')
    contact.postalAddress.administrativeArea = console_io.PromptResponse('State / Administrative area (if applicable):  ')
    contact.postalAddress.locality = console_io.PromptResponse('City / Locality:  ')
    contact.postalAddress.addressLines.append(util.PromptWithValidator(validator=util.ValidateNonEmpty, error_message=' Address Line 1 must not be empty.', prompt_string='Address Line 1:  '))
    optional_address_lines = []
    address_line_num = 2
    while len(optional_address_lines) < 4:
        address_line_num = 2 + len(optional_address_lines)
        address_line = console_io.PromptResponse('Address Line {} (if applicable):  '.format(address_line_num))
        if not address_line:
            break
        optional_address_lines += [address_line]
    if optional_address_lines:
        contact.postalAddress.addressLines.extend(optional_address_lines)
    return contact