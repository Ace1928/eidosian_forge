from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareAdminVipConfig(_messages.Message):
    """VmwareAdminVipConfig for VMware load balancer configurations.

  Fields:
    addonsVip: The VIP to configure the load balancer for add-ons.
    controlPlaneVip: The VIP which you previously set aside for the Kubernetes
      API of the admin cluster.
  """
    addonsVip = _messages.StringField(1)
    controlPlaneVip = _messages.StringField(2)