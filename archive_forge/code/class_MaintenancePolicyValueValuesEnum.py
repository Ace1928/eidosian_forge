from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MaintenancePolicyValueValuesEnum(_messages.Enum):
    """Specifies how to handle instances when a node in the group undergoes
    maintenance. Set to one of: DEFAULT, RESTART_IN_PLACE, or
    MIGRATE_WITHIN_NODE_GROUP. The default value is DEFAULT. For more
    information, see Maintenance policies.

    Values:
      DEFAULT: Allow the node and corresponding instances to retain default
        maintenance behavior.
      MAINTENANCE_POLICY_UNSPECIFIED: <no description>
      MIGRATE_WITHIN_NODE_GROUP: When maintenance must be done on a node, the
        instances on that node will be moved to other nodes in the group.
        Instances with onHostMaintenance = MIGRATE will live migrate to their
        destinations while instances with onHostMaintenance = TERMINATE will
        terminate and then restart on their destination nodes if
        automaticRestart = true.
      RESTART_IN_PLACE: Instances in this group will restart on the same node
        when maintenance has completed. Instances must have onHostMaintenance
        = TERMINATE, and they will only restart if automaticRestart = true.
    """
    DEFAULT = 0
    MAINTENANCE_POLICY_UNSPECIFIED = 1
    MIGRATE_WITHIN_NODE_GROUP = 2
    RESTART_IN_PLACE = 3