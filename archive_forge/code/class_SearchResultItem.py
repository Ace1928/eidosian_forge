from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchResultItem(_messages.Message):
    """Encapsulates each piece of data that matches the search query.

  Fields:
    asset: The resource name of the asset in the form:
      'projects/{project}/locations/{location}/assetTypes/{type}/assets/{asset
      }'
    segments: Matching segments within the above asset.
  """
    asset = _messages.StringField(1)
    segments = _messages.MessageField('Segment', 2, repeated=True)