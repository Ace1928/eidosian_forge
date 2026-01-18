from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdditionalNodeNetworkConfig(_messages.Message):
    """AdditionalNodeNetworkConfig is the configuration for additional node
  networks within the NodeNetworkConfig message

  Fields:
    network: Name of the VPC where the additional interface belongs
    subnetwork: Name of the subnetwork where the additional interface belongs
  """
    network = _messages.StringField(1)
    subnetwork = _messages.StringField(2)