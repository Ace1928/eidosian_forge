from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainsProjectsLocationsRegistrationsDeleteRequest(_messages.Message):
    """A DomainsProjectsLocationsRegistrationsDeleteRequest object.

  Fields:
    name: Required. The name of the `Registration` to delete, in the format
      `projects/*/locations/*/registrations/*`.
  """
    name = _messages.StringField(1, required=True)