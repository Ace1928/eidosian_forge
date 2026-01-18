from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExportResourcesRequest(_messages.Message):
    """Request to export resources.

  Fields:
    _since: If provided, only resources updated after this time are exported.
      The time uses the format YYYY-MM-DDThh:mm:ss.sss+zz:zz. For example,
      `2015-02-07T13:28:17.239+02:00` or `2017-01-01T00:00:00Z`. The time must
      be specified to the second and include a time zone.
    _type: String of comma-delimited FHIR resource types. If provided, only
      resources of the specified resource type(s) are exported.
    bigqueryDestination: The BigQuery output destination. The Cloud Healthcare
      Service Agent requires two IAM roles on the BigQuery location:
      `roles/bigquery.dataEditor` and `roles/bigquery.jobUser`. The output is
      one BigQuery table per resource type. Unlike when setting
      `BigQueryDestination` for `StreamConfig`, `ExportResources` does not
      create BigQuery views.
    gcsDestination: The Cloud Storage output destination. The Healthcare
      Service Agent account requires the `roles/storage.objectAdmin` role on
      the Cloud Storage location. The exported outputs are organized by FHIR
      resource types. The server creates one or more objects per resource type
      depending on the volume of the resources exported. When there is only
      one object per resource type, the object name is in the form of
      `{operation_id}_{resource_type}`. When there are multiple objects for a
      given resource type, the object names are in the form of
      `{operation_id}_{resource_type}-{index}-of-{total}`. Each object
      contains newline delimited JSON, and each line is a FHIR resource.
  """
    _since = _messages.StringField(1)
    _type = _messages.StringField(2)
    bigqueryDestination = _messages.MessageField('GoogleCloudHealthcareV1alpha2FhirBigQueryDestination', 3)
    gcsDestination = _messages.MessageField('GoogleCloudHealthcareV1alpha2FhirGcsDestination', 4)