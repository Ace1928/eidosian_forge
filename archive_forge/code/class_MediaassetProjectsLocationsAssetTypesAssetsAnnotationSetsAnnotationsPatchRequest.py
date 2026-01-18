from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsPatchRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsPa
  tchRequest object.

  Fields:
    annotation: A Annotation resource to be passed as the request body.
    name: Output only. An automatically-generated resource name of the
      annotation `projects/{project}/locations/{location}/assetTypes/{asset_ty
      pe}/assets/{asset}/annotationSets/{annotation_set}/annotations/{annotati
      on}`.
    updateMask: Required. Comma-separated list of fields to be updated.
  """
    annotation = _messages.MessageField('Annotation', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)