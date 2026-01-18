from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DistributionPolicy(_messages.Message):
    """A DistributionPolicy object.

  Enums:
    TargetShapeValueValuesEnum: The distribution shape to which the group
      converges either proactively or on resize events (depending on the value
      set in updatePolicy.instanceRedistributionType).

  Fields:
    targetShape: The distribution shape to which the group converges either
      proactively or on resize events (depending on the value set in
      updatePolicy.instanceRedistributionType).
    zones: Zones where the regional managed instance group will create and
      manage its instances.
  """

    class TargetShapeValueValuesEnum(_messages.Enum):
        """The distribution shape to which the group converges either proactively
    or on resize events (depending on the value set in
    updatePolicy.instanceRedistributionType).

    Values:
      ANY: The group picks zones for creating VM instances to fulfill the
        requested number of VMs within present resource constraints and to
        maximize utilization of unused zonal reservations. Recommended for
        batch workloads that do not require high availability.
      ANY_SINGLE_ZONE: The group creates all VM instances within a single
        zone. The zone is selected based on the present resource constraints
        and to maximize utilization of unused zonal reservations. Recommended
        for batch workloads with heavy interprocess communication.
      BALANCED: The group prioritizes acquisition of resources, scheduling VMs
        in zones where resources are available while distributing VMs as
        evenly as possible across selected zones to minimize the impact of
        zonal failure. Recommended for highly available serving workloads.
      EVEN: The group schedules VM instance creation and deletion to achieve
        and maintain an even number of managed instances across the selected
        zones. The distribution is even when the number of managed instances
        does not differ by more than 1 between any two zones. Recommended for
        highly available serving workloads.
    """
        ANY = 0
        ANY_SINGLE_ZONE = 1
        BALANCED = 2
        EVEN = 3
    targetShape = _messages.EnumField('TargetShapeValueValuesEnum', 1)
    zones = _messages.MessageField('DistributionPolicyZoneConfiguration', 2, repeated=True)