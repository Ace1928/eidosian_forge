from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetShapeValueValuesEnum(_messages.Enum):
    """Strategy for distributing VMs across zones in a region.

    Values:
      ANY: GCE picks zones for creating VM instances to fulfill the requested
        number of VMs within present resource constraints and to maximize
        utilization of unused zonal reservations. Recommended for batch
        workloads that do not require high availability.
      ANY_SINGLE_ZONE: GCE always selects a single zone for all the VMs,
        optimizing for resource quotas, available reservations and general
        capacity. Recommended for batch workloads that cannot tollerate
        distribution over multiple zones. This the default shape in Bulk
        Insert and Capacity Advisor APIs.
      BALANCED: GCE prioritizes acquisition of resources, scheduling VMs in
        zones where resources are available while distributing VMs as evenly
        as possible across allowed zones to minimize the impact of zonal
        failure. Recommended for highly available serving workloads.
    """
    ANY = 0
    ANY_SINGLE_ZONE = 1
    BALANCED = 2