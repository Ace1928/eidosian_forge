from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalControlPlaneNodePoolConfig(_messages.Message):
    """Specifies the control plane node pool configuration.

  Fields:
    nodePoolConfig: Required. The generic configuration for a node pool
      running the control plane.
  """
    nodePoolConfig = _messages.MessageField('BareMetalNodePoolConfig', 1)