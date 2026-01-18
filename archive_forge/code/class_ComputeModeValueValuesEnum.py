from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeModeValueValuesEnum(_messages.Enum):
    """Output only. Compute mode for this stage.

    Values:
      COMPUTE_MODE_UNSPECIFIED: ComputeMode type not specified.
      BIGQUERY: This stage was processed using BigQuery slots.
      BI_ENGINE: This stage was processed using BI Engine compute.
    """
    COMPUTE_MODE_UNSPECIFIED = 0
    BIGQUERY = 1
    BI_ENGINE = 2