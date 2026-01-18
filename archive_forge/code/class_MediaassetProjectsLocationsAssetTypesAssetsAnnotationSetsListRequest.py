from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsListRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsListRequest
  object.

  Fields:
    filter: The filter to apply to list results.
    pageSize: The maximum number of annotationSets to return. The service may
      return fewer than this value. If unspecified, at most 50 annotationSets
      will be returned. The maximum value is 100; values above 100 will be
      coerced to 100.
    pageToken: A page token, received from a previous `ListAnnotationSets`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListAnnotationSets` must match the call
      that provided the page token.
    parent: Required. The name of the asset that owns this collection of
      annotationSets. Format: `projects/{project}/locations/{location}/assetTy
      pes/{asset_type}/assets/{asset}`
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)