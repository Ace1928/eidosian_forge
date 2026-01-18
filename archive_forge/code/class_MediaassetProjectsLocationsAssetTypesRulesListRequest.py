from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesRulesListRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesRulesListRequest object.

  Fields:
    pageSize: The maximum number of rules to return. The service may return
      fewer than this value. If unspecified, at most 50 books will be
      returned. The maximum value is 100; values above 100 will be coerced to
      100.
    pageToken: A page token, received from a previous `ListRules` call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to `ListRules` must match the call that provided the
      page token.
    parent: Required. The parent, which owns this collection of rules. Format:
      `projects/{project}/locations/{location}/assetTypes/{type}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)