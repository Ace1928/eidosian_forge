from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainsProjectsLocationsRegistrationsExportRequest(_messages.Message):
    """A DomainsProjectsLocationsRegistrationsExportRequest object.

  Fields:
    exportRegistrationRequest: A ExportRegistrationRequest resource to be
      passed as the request body.
    name: Required. The name of the `Registration` to export, in the format
      `projects/*/locations/*/registrations/*`.
  """
    exportRegistrationRequest = _messages.MessageField('ExportRegistrationRequest', 1)
    name = _messages.StringField(2, required=True)