from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsEntryGroupsEntriesGetRequest(_messages.Message):
    """A DatacatalogProjectsLocationsEntryGroupsEntriesGetRequest object.

  Fields:
    name: Required. The name of the entry to get.
  """
    name = _messages.StringField(1, required=True)