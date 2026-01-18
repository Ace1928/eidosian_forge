from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataGovernanceTypeValueValuesEnum(_messages.Enum):
    """Optional. If set to `DATA_MASKING`, the function is validated and made
    available as a masking function. For more information, see [Create custom
    masking routines](https://cloud.google.com/bigquery/docs/user-defined-
    functions#custom-mask).

    Values:
      DATA_GOVERNANCE_TYPE_UNSPECIFIED: The data governance type is
        unspecified.
      DATA_MASKING: The data governance type is data masking.
    """
    DATA_GOVERNANCE_TYPE_UNSPECIFIED = 0
    DATA_MASKING = 1