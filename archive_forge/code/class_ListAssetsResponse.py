from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAssetsResponse(_messages.Message):
    """ListAssets response.

  Fields:
    assets: Assets.
    nextPageToken: Token to retrieve the next page of results. It expires 72
      hours after the page token for the first page is generated. Set to empty
      if there are no remaining results.
    readTime: Time the snapshot was taken.
  """
    assets = _messages.MessageField('Asset', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    readTime = _messages.StringField(3)