from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainsProjectsLocationsRegistrationsSearchDomainsRequest(_messages.Message):
    """A DomainsProjectsLocationsRegistrationsSearchDomainsRequest object.

  Fields:
    location: Required. The location. Must be in the format
      `projects/*/locations/*`.
    query: Required. String used to search for available domain names.
  """
    location = _messages.StringField(1, required=True)
    query = _messages.StringField(2)