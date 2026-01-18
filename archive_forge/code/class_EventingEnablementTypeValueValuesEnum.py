from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventingEnablementTypeValueValuesEnum(_messages.Enum):
    """Optional. Eventing enablement type. Will be nil if eventing is not
    enabled.

    Values:
      EVENTING_ENABLEMENT_TYPE_UNSPECIFIED: Eventing Enablement Type
        Unspecifeied.
      EVENTING_AND_CONNECTION: Both connection and eventing.
      ONLY_EVENTING: Only Eventing.
    """
    EVENTING_ENABLEMENT_TYPE_UNSPECIFIED = 0
    EVENTING_AND_CONNECTION = 1
    ONLY_EVENTING = 2