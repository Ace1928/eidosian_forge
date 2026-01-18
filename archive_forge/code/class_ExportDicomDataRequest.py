from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExportDicomDataRequest(_messages.Message):
    """Exports data from the specified DICOM store. If a given resource, such
  as a DICOM object with the same SOPInstance UID, already exists in the
  output, it is overwritten with the version in the source dataset. Exported
  DICOM data persists when the DICOM store from which it was exported is
  deleted.

  Fields:
    bigqueryDestination: The BigQuery output destination. You can only export
      to a BigQuery dataset that's in the same project as the DICOM store
      you're exporting from. The Cloud Healthcare Service Agent requires two
      IAM roles on the BigQuery location: `roles/bigquery.dataEditor` and
      `roles/bigquery.jobUser`.
    filterConfig: Specifies the filter configuration.
    gcsDestination: The Cloud Storage output destination. The Cloud Healthcare
      Service Agent requires the `roles/storage.objectAdmin` Cloud IAM roles
      on the Cloud Storage location.
  """
    bigqueryDestination = _messages.MessageField('GoogleCloudHealthcareV1alpha2DicomBigQueryDestination', 1)
    filterConfig = _messages.MessageField('DicomFilterConfig', 2)
    gcsDestination = _messages.MessageField('GoogleCloudHealthcareV1alpha2DicomGcsDestination', 3)