from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PlanningStatusValueValuesEnum(_messages.Enum):
    """Planning state before being submitted for evaluation

    Values:
      DRAFT: Future Reservation is being drafted.
      PLANNING_STATUS_UNSPECIFIED: <no description>
      SUBMITTED: Future Reservation has been submitted for evaluation by GCP.
    """
    DRAFT = 0
    PLANNING_STATUS_UNSPECIFIED = 1
    SUBMITTED = 2