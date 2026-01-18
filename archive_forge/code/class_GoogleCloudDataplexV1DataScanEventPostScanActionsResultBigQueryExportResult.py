from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataScanEventPostScanActionsResultBigQueryExportResult(_messages.Message):
    """The result of BigQuery export post scan action.

  Enums:
    StateValueValuesEnum: Execution state for the BigQuery exporting.

  Fields:
    message: Additional information about the BigQuery exporting.
    state: Execution state for the BigQuery exporting.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Execution state for the BigQuery exporting.

    Values:
      STATE_UNSPECIFIED: The exporting state is unspecified.
      SUCCEEDED: The exporting completed successfully.
      FAILED: The exporting is no longer running due to an error.
      SKIPPED: The exporting is skipped due to no valid scan result to export
        (usually caused by scan failed).
    """
        STATE_UNSPECIFIED = 0
        SUCCEEDED = 1
        FAILED = 2
        SKIPPED = 3
    message = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)