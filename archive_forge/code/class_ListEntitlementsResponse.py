from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListEntitlementsResponse(_messages.Message):
    """Message for response to listing Entitlements.

  Fields:
    entitlements: The list of Entitlements.
    nextPageToken: A token identifying a page of results the server should
      return.
    unreachable: Locations that could not be reached.
  """
    entitlements = _messages.MessageField('Entitlement', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)