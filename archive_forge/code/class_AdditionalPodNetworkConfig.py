from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdditionalPodNetworkConfig(_messages.Message):
    """AdditionalPodNetworkConfig is the configuration for additional pod
  networks within the NodeNetworkConfig message

  Fields:
    maxPodsPerNode: The maximum number of pods per node which use this pod
      network
    secondaryPodRange: The name of the secondary range on the subnet which
      provides IP address for this pod range
    subnetwork: Name of the subnetwork where the additional pod network
      belongs
  """
    maxPodsPerNode = _messages.MessageField('MaxPodsConstraint', 1)
    secondaryPodRange = _messages.StringField(2)
    subnetwork = _messages.StringField(3)