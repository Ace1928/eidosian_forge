from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudNatStatus(_messages.Message):
    """CloudNatStatus contains the desired state of the cloud nat functionality
  on this cluster.

  Fields:
    enabled: Enables Cloud Nat on this cluster. On an update if
      update.desired_cloud_nat_status.enabled = true, The API will check if
      any Routers in the cluster's network has Cloud NAT enabled on the pod
      range. a. If so, then the cluster nodes will be updated to not perform
      SNAT. b. If no NAT configuration exists, a new Router with Cloud NAT on
      the secondary range will be created first, and then the nodes will be
      updated to no longer do SNAT.
  """
    enabled = _messages.BooleanField(1)