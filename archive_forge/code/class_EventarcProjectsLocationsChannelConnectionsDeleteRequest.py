from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventarcProjectsLocationsChannelConnectionsDeleteRequest(_messages.Message):
    """A EventarcProjectsLocationsChannelConnectionsDeleteRequest object.

  Fields:
    name: Required. The name of the channel connection to delete.
  """
    name = _messages.StringField(1, required=True)