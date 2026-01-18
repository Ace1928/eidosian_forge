from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesTableListRequest(_messages.Message):
    """A FusiontablesTableListRequest object.

  Fields:
    maxResults: Maximum number of styles to return. Optional. Default is 5.
    pageToken: Continuation token specifying which result page to return.
      Optional.
  """
    maxResults = _messages.IntegerField(1, variant=_messages.Variant.UINT32)
    pageToken = _messages.StringField(2)