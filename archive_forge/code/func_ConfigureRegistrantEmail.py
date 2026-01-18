from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def ConfigureRegistrantEmail(self, registration_ref, registrant_email):
    """Sets a registrant contact.

    This resends registrant email confirmation.
    It's done by updating registrant email to the current value.

    Args:
      registration_ref: a Resource reference to a
        domains.projects.locations.registrations resource.
      registrant_email: The registrant email.

    Returns:
      Operation: the long running operation to configure contacts registration.
    """
    contact_settings = self.messages.ContactSettings(registrantContact=self.messages.Contact(email=registrant_email))
    req = self.messages.DomainsProjectsLocationsRegistrationsConfigureContactSettingsRequest(registration=registration_ref.RelativeName(), configureContactSettingsRequest=self.messages.ConfigureContactSettingsRequest(contactSettings=contact_settings, updateMask='registrant_contact.email'))
    return self._service.ConfigureContactSettings(req)