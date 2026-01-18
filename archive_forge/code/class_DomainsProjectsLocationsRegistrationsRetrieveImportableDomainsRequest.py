from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainsProjectsLocationsRegistrationsRetrieveImportableDomainsRequest(_messages.Message):
    """A DomainsProjectsLocationsRegistrationsRetrieveImportableDomainsRequest
  object.

  Fields:
    location: Required. The location. Must be in the format
      `projects/*/locations/*`.
    pageSize: Maximum number of results to return.
    pageToken: When set to the `next_page_token` from a prior response,
      provides the next page of results.
  """
    location = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)