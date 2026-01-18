from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareDataplaneV2Config(_messages.Message):
    """Contains configurations for Dataplane V2, which is optimized dataplane
  for Kubernetes networking. For more information, see:
  https://cloud.google.com/kubernetes-engine/docs/concepts/dataplane-v2

  Fields:
    advancedNetworking: Enable advanced networking which requires
      dataplane_v2_enabled to be set true.
    dataplaneV2Enabled: Enables Dataplane V2.
    forwardMode: Configure ForwardMode for Dataplane v2.
    windowsDataplaneV2Enabled: Enable Dataplane V2 for clusters with Windows
      nodes.
  """
    advancedNetworking = _messages.BooleanField(1)
    dataplaneV2Enabled = _messages.BooleanField(2)
    forwardMode = _messages.StringField(3)
    windowsDataplaneV2Enabled = _messages.BooleanField(4)