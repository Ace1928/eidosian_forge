from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTensorFlowVersionsResponse(_messages.Message):
    """Response for ListTensorFlowVersions.

  Fields:
    nextPageToken: The next page token or empty if none.
    tensorflowVersions: The listed nodes.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    tensorflowVersions = _messages.MessageField('TensorFlowVersion', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)