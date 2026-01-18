from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataScanJob(_messages.Message):
    """A DataScanJob represents an instance of DataScan execution.

  Enums:
    StateValueValuesEnum: Output only. Execution state for the DataScanJob.
    TypeValueValuesEnum: Output only. The type of the parent DataScan.

  Fields:
    dataProfileResult: Output only. The result of the data profile scan.
    dataProfileSpec: Output only. DataProfileScan related setting.
    dataQualityResult: Output only. The result of the data quality scan.
    dataQualitySpec: Output only. DataQualityScan related setting.
    endTime: Output only. The time when the DataScanJob ended.
    message: Output only. Additional information about the current state.
    name: Output only. The relative resource name of the DataScanJob, of the
      form: projects/{project}/locations/{location_id}/dataScans/{datascan_id}
      /jobs/{job_id}, where project refers to a project_id or project_number
      and location_id refers to a GCP region.
    startTime: Output only. The time when the DataScanJob was started.
    state: Output only. Execution state for the DataScanJob.
    type: Output only. The type of the parent DataScan.
    uid: Output only. System generated globally unique ID for the DataScanJob.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Execution state for the DataScanJob.

    Values:
      STATE_UNSPECIFIED: The DataScanJob state is unspecified.
      RUNNING: The DataScanJob is running.
      CANCELING: The DataScanJob is canceling.
      CANCELLED: The DataScanJob cancellation was successful.
      SUCCEEDED: The DataScanJob completed successfully.
      FAILED: The DataScanJob is no longer running due to an error.
      PENDING: The DataScanJob has been created but not started to run yet.
    """
        STATE_UNSPECIFIED = 0
        RUNNING = 1
        CANCELING = 2
        CANCELLED = 3
        SUCCEEDED = 4
        FAILED = 5
        PENDING = 6

    class TypeValueValuesEnum(_messages.Enum):
        """Output only. The type of the parent DataScan.

    Values:
      DATA_SCAN_TYPE_UNSPECIFIED: The DataScan type is unspecified.
      DATA_QUALITY: Data Quality scan.
      DATA_PROFILE: Data Profile scan.
    """
        DATA_SCAN_TYPE_UNSPECIFIED = 0
        DATA_QUALITY = 1
        DATA_PROFILE = 2
    dataProfileResult = _messages.MessageField('GoogleCloudDataplexV1DataProfileResult', 1)
    dataProfileSpec = _messages.MessageField('GoogleCloudDataplexV1DataProfileSpec', 2)
    dataQualityResult = _messages.MessageField('GoogleCloudDataplexV1DataQualityResult', 3)
    dataQualitySpec = _messages.MessageField('GoogleCloudDataplexV1DataQualitySpec', 4)
    endTime = _messages.StringField(5)
    message = _messages.StringField(6)
    name = _messages.StringField(7)
    startTime = _messages.StringField(8)
    state = _messages.EnumField('StateValueValuesEnum', 9)
    type = _messages.EnumField('TypeValueValuesEnum', 10)
    uid = _messages.StringField(11)