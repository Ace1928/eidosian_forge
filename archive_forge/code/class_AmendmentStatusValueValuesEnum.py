from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AmendmentStatusValueValuesEnum(_messages.Enum):
    """[Output Only] The current status of the requested amendment.

    Values:
      AMENDMENT_APPROVED: The requested amendment to the Future Resevation has
        been approved and applied by GCP.
      AMENDMENT_DECLINED: The requested amendment to the Future Reservation
        has been declined by GCP and the original state was restored.
      AMENDMENT_IN_REVIEW: The requested amendment to the Future Reservation
        is currently being reviewd by GCP.
      AMENDMENT_STATUS_UNSPECIFIED: <no description>
    """
    AMENDMENT_APPROVED = 0
    AMENDMENT_DECLINED = 1
    AMENDMENT_IN_REVIEW = 2
    AMENDMENT_STATUS_UNSPECIFIED = 3