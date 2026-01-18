from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1NearestNeighborQueryStringFilter(_messages.Message):
    """String filter is used to search a subset of the entities by using
  boolean rules on string columns. For example: if a query specifies string
  filter with 'name = color, allow_tokens = {red, blue}, deny_tokens =
  {purple}',' then that query will match entities that are red or blue, but if
  those points are also purple, then they will be excluded even if they are
  red/blue. Only string filter is supported for now, numeric filter will be
  supported in the near future.

  Fields:
    allowTokens: Optional. The allowed tokens.
    denyTokens: Optional. The denied tokens.
    name: Required. Column names in BigQuery that used as filters.
  """
    allowTokens = _messages.StringField(1, repeated=True)
    denyTokens = _messages.StringField(2, repeated=True)
    name = _messages.StringField(3)