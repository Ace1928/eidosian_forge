from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsListRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsLi
  stRequest object.

  Fields:
    filter: The filter to apply to list results. Valid field expressions are
      defined in assetType.annotationSetConfig.indexedFieldConfig. Format:
      https://cloud.google.com/logging/docs/view/advanced-queries
    pageSize: The maximum number of annotations to return. The service may
      return fewer than this value. If unspecified, at most 50 annotations
      will be returned. The maximum value is 100; values above 100 will be
      coerced to 100.
    pageToken: A page token, received from a previous `ListAnnotations` call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to `ListAnnotations` must match the call that
      provided the page token.
    parent: Required. The parent, which owns this collection of annotations.
      Format: `projects/{project}/locations/{location}/assetTypes/{asset_type}
      /assets/{asset}/annotationSets/{annotation_set}`
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)