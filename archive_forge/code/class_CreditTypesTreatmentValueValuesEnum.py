from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreditTypesTreatmentValueValuesEnum(_messages.Enum):
    """Optional. If not set, default behavior is `INCLUDE_ALL_CREDITS`.

    Values:
      CREDIT_TYPES_TREATMENT_UNSPECIFIED: <no description>
      INCLUDE_ALL_CREDITS: All types of credit are subtracted from the gross
        cost to determine the spend for threshold calculations.
      EXCLUDE_ALL_CREDITS: All types of credit are added to the net cost to
        determine the spend for threshold calculations.
      INCLUDE_SPECIFIED_CREDITS: [Credit
        types](https://cloud.google.com/billing/docs/how-to/export-data-
        bigquery-tables#credits-type) specified in the credit_types field are
        subtracted from the gross cost to determine the spend for threshold
        calculations.
    """
    CREDIT_TYPES_TREATMENT_UNSPECIFIED = 0
    INCLUDE_ALL_CREDITS = 1
    EXCLUDE_ALL_CREDITS = 2
    INCLUDE_SPECIFIED_CREDITS = 3