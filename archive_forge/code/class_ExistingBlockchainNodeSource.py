from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExistingBlockchainNodeSource(_messages.Message):
    """Configuration for deploying blockchain validators to an existing
  blockchain node.

  Fields:
    blockchainNodeId: Optional. Name of the blockchain node to deploy the
      validators to. If not set, the validators are not deployed.
  """
    blockchainNodeId = _messages.StringField(1)