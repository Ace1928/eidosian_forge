from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceManagementValueValuesEnum(_messages.Enum):
    """ServiceManagementValueValuesEnum enum type.

    Values:
      SERVICE_MANAGEMENT_UNKNOWN_REASON: An unknown reason indicates that we
        have not received a signal from service management about this
        container. Since containers are created by request of service
        management, this reason should never be set.
      SERVICE_MANAGEMENT_CONTROL_PLANE_SYNC: Due to various reasons CCFE might
        proactively restate a container state to a CLH to ensure that the CLH
        and CCFE are both aware of the container state. This reason can be
        tied to any of the states.
      ACTIVATION: When a customer activates an API CCFE notifies the CLH and
        sets the container to the ON state.
      PREPARE_DEACTIVATION: When a customer deactivates and API service
        management starts a two-step process to perform the deactivation. The
        first step is to prepare. Prepare is a reason to put the container in
        a EXTERNAL_OFF state.
      ABORT_DEACTIVATION: If the deactivation is cancelled, service managed
        needs to abort the deactivation. Abort is a reason to put the
        container in an ON state.
      COMMIT_DEACTIVATION: If the deactivation is followed through with,
        service management needs to finish deactivation. Commit is a reason to
        put the container in a DELETED state.
    """
    SERVICE_MANAGEMENT_UNKNOWN_REASON = 0
    SERVICE_MANAGEMENT_CONTROL_PLANE_SYNC = 1
    ACTIVATION = 2
    PREPARE_DEACTIVATION = 3
    ABORT_DEACTIVATION = 4
    COMMIT_DEACTIVATION = 5