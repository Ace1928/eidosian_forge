from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListGcpUserAccessBindingsResponse(_messages.Message):
    """Response of ListGcpUserAccessBindings.

  Fields:
    gcpUserAccessBindings: GcpUserAccessBinding
    nextPageToken: Token to get the next page of items. If blank, there are no
      more items.
  """
    gcpUserAccessBindings = _messages.MessageField('GcpUserAccessBinding', 1, repeated=True)
    nextPageToken = _messages.StringField(2)