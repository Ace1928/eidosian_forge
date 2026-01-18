from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchEntitlementsResponse(_messages.Message):
    """Response message for `SearchEntitlements` method.

  Fields:
    entitlements: The list of Entitlements.
    nextPageToken: A token identifying a page of results the server should
      return.
  """
    entitlements = _messages.MessageField('Entitlement', 1, repeated=True)
    nextPageToken = _messages.StringField(2)