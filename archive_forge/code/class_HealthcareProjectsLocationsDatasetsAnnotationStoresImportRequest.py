from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsAnnotationStoresImportRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsAnnotationStoresImportRequest
  object.

  Fields:
    importAnnotationsRequest: A ImportAnnotationsRequest resource to be passed
      as the request body.
    name: Required. The name of the Annotation store to which the server
      imports annotations in the format of `projects/{project_id}/locations/{l
      ocation_id}/datasets/{dataset_id}/annotationStores/{annotation_store_id}
      `.
  """
    importAnnotationsRequest = _messages.MessageField('ImportAnnotationsRequest', 1)
    name = _messages.StringField(2, required=True)