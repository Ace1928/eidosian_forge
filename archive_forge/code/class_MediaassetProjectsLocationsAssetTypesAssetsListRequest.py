from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesAssetsListRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesAssetsListRequest object.

  Fields:
    filter: The filter to apply to list results. Valid field expressions are
      defined in assetType.indexedFieldConfig. Format:
      https://cloud.google.com/logging/docs/view/advanced-queries
    pageSize: The maximum number of items to return. If unspecified, server
      will pick an appropriate default. Server may return fewer items than
      requested. A caller should only rely on response's next_page_token to
      determine if there are more realms left to be queried.
    pageToken: The next_page_token value returned from a previous List
      request, if any.
    parent: Required. The parent resource name, in the following form:
      `projects/{project}/locations/{location}/assetTypes/{type}`.
    readMask: Extra fields to be poplulated as part of the asset resource in
      the response. Currently, this only supports populating asset metadata
      (no wildcards and no contents of the entire asset).
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    readMask = _messages.StringField(5)