from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainsProjectsLocationsRegistrationsConfigureManagementSettingsRequest(_messages.Message):
    """A
  DomainsProjectsLocationsRegistrationsConfigureManagementSettingsRequest
  object.

  Fields:
    configureManagementSettingsRequest: A ConfigureManagementSettingsRequest
      resource to be passed as the request body.
    registration: Required. The name of the `Registration` whose management
      settings are being updated, in the format
      `projects/*/locations/*/registrations/*`.
  """
    configureManagementSettingsRequest = _messages.MessageField('ConfigureManagementSettingsRequest', 1)
    registration = _messages.StringField(2, required=True)