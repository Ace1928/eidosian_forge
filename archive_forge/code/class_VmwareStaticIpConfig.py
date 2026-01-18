from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareStaticIpConfig(_messages.Message):
    """Represents the network configuration required for the VMware user
  clusters with Static IP configurations.

  Fields:
    ipBlocks: Represents the configuration values for static IP allocation to
      nodes.
  """
    ipBlocks = _messages.MessageField('VmwareIpBlock', 1, repeated=True)