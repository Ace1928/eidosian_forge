from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainsProjectsLocationsRegistrationsConfigureDnsSettingsRequest(_messages.Message):
    """A DomainsProjectsLocationsRegistrationsConfigureDnsSettingsRequest
  object.

  Fields:
    configureDnsSettingsRequest: A ConfigureDnsSettingsRequest resource to be
      passed as the request body.
    registration: Required. The name of the `Registration` whose DNS settings
      are being updated, in the format
      `projects/*/locations/*/registrations/*`.
  """
    configureDnsSettingsRequest = _messages.MessageField('ConfigureDnsSettingsRequest', 1)
    registration = _messages.StringField(2, required=True)