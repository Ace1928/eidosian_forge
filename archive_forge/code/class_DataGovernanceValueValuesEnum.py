from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataGovernanceValueValuesEnum(_messages.Enum):
    """DataGovernanceValueValuesEnum enum type.

    Values:
      DATA_GOVERNANCE_UNKNOWN_REASON: An unknown reason indicates that data
        governance has not sent a signal for this container.
      DATA_GOVERNANCE_CONTROL_PLANE_SYNC: Due to various reasons CCFE might
        proactively restate a container state to a CLH to ensure that the CLH
        and CCFE are both aware of the container state. This reason can be
        tied to any of the states.
      HIDE: When a container is deleted we retain some data for a period of
        time to allow the consumer to change their mind. Data governance sends
        a signal to hide the data when this occurs. Hide is a reason to put
        the container in an INTERNAL_OFF state.
      UNHIDE: The decision to un-delete a container can be made. When this
        happens data governance tells us to unhide any hidden data. Unhide is
        a reason to put the container in an ON state.
      PURGE: After a period of time data must be completely removed from our
        systems. When data governance sends a purge signal we need to remove
        data. Purge is a reason to put the container in a DELETED state. Purge
        is the only event that triggers a delete mutation. All other events
        have update semantics.
    """
    DATA_GOVERNANCE_UNKNOWN_REASON = 0
    DATA_GOVERNANCE_CONTROL_PLANE_SYNC = 1
    HIDE = 2
    UNHIDE = 3
    PURGE = 4