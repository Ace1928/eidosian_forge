from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainsProjectsLocationsRegistrationsConfigureContactSettingsRequest(_messages.Message):
    """A DomainsProjectsLocationsRegistrationsConfigureContactSettingsRequest
  object.

  Fields:
    configureContactSettingsRequest: A ConfigureContactSettingsRequest
      resource to be passed as the request body.
    registration: Required. The name of the `Registration` whose contact
      settings are being updated, in the format
      `projects/*/locations/*/registrations/*`.
  """
    configureContactSettingsRequest = _messages.MessageField('ConfigureContactSettingsRequest', 1)
    registration = _messages.StringField(2, required=True)