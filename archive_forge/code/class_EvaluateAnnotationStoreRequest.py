from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EvaluateAnnotationStoreRequest(_messages.Message):
    """Request to evaluate an Annotation store against a ground truth
  [Annotation store].

  Fields:
    bigqueryDestination: The BigQuery table where the server writes the
      output. BigQueryDestination requires the `roles/bigquery.dataEditor` and
      `roles/bigquery.jobUser` Cloud IAM roles.
    goldenStore: Required. The Annotation store to use as ground truth, in the
      format of `projects/{project_id}/locations/{location_id}/datasets/{datas
      et_id}/annotationStores/{annotation_store_id}`.
  """
    bigqueryDestination = _messages.MessageField('GoogleCloudHealthcareV1alpha2AnnotationBigQueryDestination', 1)
    goldenStore = _messages.StringField(2)