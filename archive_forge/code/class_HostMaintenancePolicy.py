from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HostMaintenancePolicy(_messages.Message):
    """HostMaintenancePolicy contains the maintenance policy for the hosts on
  which the GKE VMs run on.

  Enums:
    MaintenanceIntervalValueValuesEnum: Specifies the frequency of planned
      maintenance events.

  Fields:
    maintenanceInterval: Specifies the frequency of planned maintenance
      events.
    opportunisticMaintenanceStrategy: Strategy that will trigger maintenance
      on behalf of the customer.
  """

    class MaintenanceIntervalValueValuesEnum(_messages.Enum):
        """Specifies the frequency of planned maintenance events.

    Values:
      MAINTENANCE_INTERVAL_UNSPECIFIED: The maintenance interval is not
        explicitly specified.
      AS_NEEDED: Nodes are eligible to receive infrastructure and hypervisor
        updates as they become available. This may result in more maintenance
        operations (live migrations or terminations) for the node than the
        PERIODIC option.
      PERIODIC: Nodes receive infrastructure and hypervisor updates on a
        periodic basis, minimizing the number of maintenance operations (live
        migrations or terminations) on an individual VM. This may mean
        underlying VMs will take longer to receive an update than if it was
        configured for AS_NEEDED. Security updates will still be applied as
        soon as they are available.
    """
        MAINTENANCE_INTERVAL_UNSPECIFIED = 0
        AS_NEEDED = 1
        PERIODIC = 2
    maintenanceInterval = _messages.EnumField('MaintenanceIntervalValueValuesEnum', 1)
    opportunisticMaintenanceStrategy = _messages.MessageField('OpportunisticMaintenanceStrategy', 2)