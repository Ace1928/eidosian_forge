from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsEntryGroupsGetRequest(_messages.Message):
    """A DatacatalogProjectsLocationsEntryGroupsGetRequest object.

  Fields:
    name: Required. The name of the entry group to get.
    readMask: The fields to return. If empty or omitted, all fields are
      returned.
  """
    name = _messages.StringField(1, required=True)
    readMask = _messages.StringField(2)