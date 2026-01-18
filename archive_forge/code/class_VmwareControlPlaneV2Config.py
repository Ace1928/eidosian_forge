from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareControlPlaneV2Config(_messages.Message):
    """Specifies control plane V2 config.

  Fields:
    controlPlaneIpBlock: Static IP addresses for the control plane nodes.
  """
    controlPlaneIpBlock = _messages.MessageField('VmwareIpBlock', 1)