from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalListNodesResponse(_messages.Message):
    """Response for ListNodes.

  Fields:
    nextPageToken: A pagination token returned from a previous call to
      ListNodes that indicates from where listing should continue. If the
      field is missing or empty, it means there is no more nodes.
    nodes: The nodes that match the request.
  """
    nextPageToken = _messages.StringField(1)
    nodes = _messages.MessageField('SasPortalNode', 2, repeated=True)