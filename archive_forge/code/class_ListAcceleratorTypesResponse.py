from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAcceleratorTypesResponse(_messages.Message):
    """Response for ListAcceleratorTypes.

  Fields:
    acceleratorTypes: The listed nodes.
    nextPageToken: The next page token or empty if none.
    unreachable: Locations that could not be reached.
  """
    acceleratorTypes = _messages.MessageField('AcceleratorType', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)