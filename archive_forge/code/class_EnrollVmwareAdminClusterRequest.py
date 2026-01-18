from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnrollVmwareAdminClusterRequest(_messages.Message):
    """Message for enrolling an existing VMware admin cluster to the GKE on-
  prem API.

  Fields:
    membership: Required. This is the full resource name of this admin
      cluster's fleet membership.
    vmwareAdminClusterId: User provided OnePlatform identifier that is used as
      part of the resource name. This must be unique among all GKE on-prem
      clusters within a project and location and will return a 409 if the
      cluster already exists. (https://tools.ietf.org/html/rfc1123) format.
  """
    membership = _messages.StringField(1)
    vmwareAdminClusterId = _messages.StringField(2)