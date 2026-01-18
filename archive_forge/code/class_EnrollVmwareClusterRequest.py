from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnrollVmwareClusterRequest(_messages.Message):
    """Message for enrolling an existing VMware cluster to the Anthos On-Prem
  API.

  Fields:
    adminClusterMembership: Required. The admin cluster this VMware user
      cluster belongs to. This is the full resource name of the admin
      cluster's fleet membership. In the future, references to other resource
      types might be allowed if admin clusters are modeled as their own
      resources.
    localName: Optional. The object name of the VMware OnPremUserCluster
      custom resource on the associated admin cluster. This field is used to
      support conflicting resource names when enrolling existing clusters to
      the API. When not provided, this field will resolve to the
      vmware_cluster_id. Otherwise, it must match the object name of the
      VMware OnPremUserCluster custom resource. It is not modifiable outside /
      beyond the enrollment operation.
    validateOnly: Validate the request without actually doing any updates.
    vmwareClusterId: User provided OnePlatform identifier that is used as part
      of the resource name. This must be unique among all GKE on-prem clusters
      within a project and location and will return a 409 if the cluster
      already exists. (https://tools.ietf.org/html/rfc1123) format.
  """
    adminClusterMembership = _messages.StringField(1)
    localName = _messages.StringField(2)
    validateOnly = _messages.BooleanField(3)
    vmwareClusterId = _messages.StringField(4)