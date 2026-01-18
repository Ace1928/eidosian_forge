from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsAnnotationStoresPatchRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsAnnotationStoresPatchRequest
  object.

  Fields:
    annotationStore: A AnnotationStore resource to be passed as the request
      body.
    name: Identifier. Resource name of the Annotation store, of the form `proj
      ects/{project_id}/locations/{location_id}/datasets/{dataset_id}/annotati
      onStores/{annotation_store_id}`.
    updateMask: Required. The update mask applies to the resource. For the
      `FieldMask` definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask
  """
    annotationStore = _messages.MessageField('AnnotationStore', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)