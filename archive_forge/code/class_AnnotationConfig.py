from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnnotationConfig(_messages.Message):
    """Specifies how to store annotations during de-identification operations.

  Fields:
    annotationStoreName: The name of the annotation store, in the form `projec
      ts/{project_id}/locations/{location_id}/datasets/{dataset_id}/annotation
      Stores/{annotation_store_id}`. * The destination annotation store must
      be in the same location as the source data. De-identifying data across
      multiple locations is not supported. * The destination annotation store
      must exist when using DeidentifyDicomStore or DeidentifyFhirStore.
      DeidentifyDataset automatically creates the destination annotation
      store.
    storeQuote: If set to true, sensitive text is included in
      SensitiveTextAnnotation of Annotation.
  """
    annotationStoreName = _messages.StringField(1)
    storeQuote = _messages.BooleanField(2)