from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ObservabilityValueValuesEnum(_messages.Enum):
    """Optional. Indicates if the user archived this incident.

    Values:
      OBSERVABILITY_UNSPECIFIED: The incident observability is unspecified.
      ACTIVE: The incident is currently active. Can change to this status from
        archived.
      ARCHIVED: The incident is currently archived and was archived by the
        customer.
    """
    OBSERVABILITY_UNSPECIFIED = 0
    ACTIVE = 1
    ARCHIVED = 2