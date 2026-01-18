from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsHttpFiltersGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsHttpFiltersGetRequest object.

  Fields:
    name: Required. A name of the HttpFilter to get. Must be in the format
      `projects/*/locations/global/httpFilters/*`.
  """
    name = _messages.StringField(1, required=True)