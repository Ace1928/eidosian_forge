from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CgroupModeValueValuesEnum(_messages.Enum):
    """cgroup_mode specifies the cgroup mode to be used on the node.

    Values:
      CGROUP_MODE_UNSPECIFIED: CGROUP_MODE_UNSPECIFIED is when unspecified
        cgroup configuration is used. The default for the GKE node OS image
        will be used.
      CGROUP_MODE_V1: CGROUP_MODE_V1 specifies to use cgroupv1 for the cgroup
        configuration on the node image.
      CGROUP_MODE_V2: CGROUP_MODE_V2 specifies to use cgroupv2 for the cgroup
        configuration on the node image.
    """
    CGROUP_MODE_UNSPECIFIED = 0
    CGROUP_MODE_V1 = 1
    CGROUP_MODE_V2 = 2