from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalAdminWorkloadNodeConfig(_messages.Message):
    """BareMetalAdminWorkloadNodeConfig specifies the workload node
  configurations.

  Fields:
    maxPodsPerNode: The maximum number of pods a node can run. The size of the
      CIDR range assigned to the node will be derived from this parameter. By
      default 110 Pods are created per Node. Upper bound is 250 for both HA
      and non-HA admin cluster. Lower bound is 64 for non-HA admin cluster and
      32 for HA admin cluster.
  """
    maxPodsPerNode = _messages.IntegerField(1)