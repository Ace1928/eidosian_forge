from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def ConfigureContacts(self, registration_ref, contacts, contact_privacy, public_contacts_ack, validate_only):
    """Calls ConfigureContactSettings method.

    Args:
      registration_ref: Registration resource reference.
      contacts: New Contacts.
      contact_privacy: New Contact privacy.
      public_contacts_ack: Whether the user accepted public privacy.
      validate_only: validate_only flag.

    Returns:
      Long Running Operation reference.
    """
    updated_list = []
    if contact_privacy:
        updated_list += ['privacy']
    if contacts is None:
        contact_settings = self.messages.ContactSettings(privacy=contact_privacy)
    else:
        contact_settings = self.messages.ContactSettings(privacy=contact_privacy, registrantContact=contacts.registrantContact, adminContact=contacts.adminContact, technicalContact=contacts.technicalContact)
        if contacts.registrantContact:
            updated_list += ['registrant_contact']
        if contacts.adminContact:
            updated_list += ['admin_contact']
        if contacts.technicalContact:
            updated_list += ['technical_contact']
    update_mask = ','.join(updated_list)
    notices = []
    if public_contacts_ack:
        notices = [self.messages.ConfigureContactSettingsRequest.ContactNoticesValueListEntryValuesEnum.PUBLIC_CONTACT_DATA_ACKNOWLEDGEMENT]
    req = self.messages.DomainsProjectsLocationsRegistrationsConfigureContactSettingsRequest(registration=registration_ref.RelativeName(), configureContactSettingsRequest=self.messages.ConfigureContactSettingsRequest(contactSettings=contact_settings, updateMask=update_mask, contactNotices=notices, validateOnly=validate_only))
    return self._service.ConfigureContactSettings(req)