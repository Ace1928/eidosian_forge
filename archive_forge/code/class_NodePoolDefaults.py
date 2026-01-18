from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodePoolDefaults(_messages.Message):
    """Subset of Nodepool message that has defaults.

  Fields:
    nodeConfigDefaults: Subset of NodeConfig message that has defaults.
  """
    nodeConfigDefaults = _messages.MessageField('NodeConfigDefaults', 1)