from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsCreateRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsCr
  eateRequest object.

  Fields:
    annotation: A Annotation resource to be passed as the request body.
    annotationId: Required. The ID of the annotation resource to be created.
    parent: Required. The parent resource where this Annotation will be
      created. Format: `projects/{project}/locations/{location}/assetTypes/{as
      set_type}/assets/{asset}/annotationSets/{annotation_set}`
  """
    annotation = _messages.MessageField('Annotation', 1)
    annotationId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)