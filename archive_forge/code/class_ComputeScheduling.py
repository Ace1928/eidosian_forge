from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeScheduling(_messages.Message):
    """Scheduling information for VM on maintenance/restart behaviour and node
  allocation in sole tenant nodes.

  Enums:
    OnHostMaintenanceValueValuesEnum: How the instance should behave when the
      host machine undergoes maintenance that may temporarily impact instance
      performance.
    RestartTypeValueValuesEnum: Whether the Instance should be automatically
      restarted whenever it is terminated by Compute Engine (not terminated by
      user). This configuration is identical to `automaticRestart` field in
      Compute Engine create instance under scheduling. It was changed to an
      enum (instead of a boolean) to match the default value in Compute Engine
      which is automatic restart.

  Fields:
    minNodeCpus: The minimum number of virtual CPUs this instance will consume
      when running on a sole-tenant node. Ignored if no node_affinites are
      configured.
    nodeAffinities: A set of node affinity and anti-affinity configurations
      for sole tenant nodes.
    onHostMaintenance: How the instance should behave when the host machine
      undergoes maintenance that may temporarily impact instance performance.
    restartType: Whether the Instance should be automatically restarted
      whenever it is terminated by Compute Engine (not terminated by user).
      This configuration is identical to `automaticRestart` field in Compute
      Engine create instance under scheduling. It was changed to an enum
      (instead of a boolean) to match the default value in Compute Engine
      which is automatic restart.
  """

    class OnHostMaintenanceValueValuesEnum(_messages.Enum):
        """How the instance should behave when the host machine undergoes
    maintenance that may temporarily impact instance performance.

    Values:
      ON_HOST_MAINTENANCE_UNSPECIFIED: An unknown, unexpected behavior.
      TERMINATE: Terminate the instance when the host machine undergoes
        maintenance.
      MIGRATE: Migrate the instance when the host machine undergoes
        maintenance.
    """
        ON_HOST_MAINTENANCE_UNSPECIFIED = 0
        TERMINATE = 1
        MIGRATE = 2

    class RestartTypeValueValuesEnum(_messages.Enum):
        """Whether the Instance should be automatically restarted whenever it is
    terminated by Compute Engine (not terminated by user). This configuration
    is identical to `automaticRestart` field in Compute Engine create instance
    under scheduling. It was changed to an enum (instead of a boolean) to
    match the default value in Compute Engine which is automatic restart.

    Values:
      RESTART_TYPE_UNSPECIFIED: Unspecified behavior. This will use the
        default.
      AUTOMATIC_RESTART: The Instance should be automatically restarted
        whenever it is terminated by Compute Engine.
      NO_AUTOMATIC_RESTART: The Instance isn't automatically restarted
        whenever it is terminated by Compute Engine.
    """
        RESTART_TYPE_UNSPECIFIED = 0
        AUTOMATIC_RESTART = 1
        NO_AUTOMATIC_RESTART = 2
    minNodeCpus = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    nodeAffinities = _messages.MessageField('SchedulingNodeAffinity', 2, repeated=True)
    onHostMaintenance = _messages.EnumField('OnHostMaintenanceValueValuesEnum', 3)
    restartType = _messages.EnumField('RestartTypeValueValuesEnum', 4)