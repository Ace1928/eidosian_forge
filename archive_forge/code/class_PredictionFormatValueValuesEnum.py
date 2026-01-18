from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PredictionFormatValueValuesEnum(_messages.Enum):
    """The storage format of the predictions generated BatchPrediction job.

    Values:
      PREDICTION_FORMAT_UNSPECIFIED: Should not be set.
      JSONL: Predictions are in JSONL files.
      BIGQUERY: Predictions are in BigQuery.
    """
    PREDICTION_FORMAT_UNSPECIFIED = 0
    JSONL = 1
    BIGQUERY = 2