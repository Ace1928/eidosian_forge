from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsAnnotationStoresEvaluateRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsAnnotationStoresEvaluateRequest
  object.

  Fields:
    evaluateAnnotationStoreRequest: A EvaluateAnnotationStoreRequest resource
      to be passed as the request body.
    name: Required. The Annotation store to compare against `golden_store`, in
      the format of `projects/{project_id}/locations/{location_id}/datasets/{d
      ataset_id}/annotationStores/{annotation_store_id}`.
  """
    evaluateAnnotationStoreRequest = _messages.MessageField('EvaluateAnnotationStoreRequest', 1)
    name = _messages.StringField(2, required=True)