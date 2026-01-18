from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OnHostMaintenanceValueValuesEnum(_messages.Enum):
    """Optional. Defines the maintenance behavior for this instance.

    Values:
      ON_HOST_MAINTENANCE_UNSPECIFIED: Default value. This value is unused.
      TERMINATE: Tells Compute Engine to terminate and (optionally) restart
        the instance away from the maintenance activity.
      MIGRATE: Default, Allows Compute Engine to automatically migrate
        instances out of the way of maintenance events.
    """
    ON_HOST_MAINTENANCE_UNSPECIFIED = 0
    TERMINATE = 1
    MIGRATE = 2