from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NewBlockchainNodeSource(_messages.Message):
    """Configuration for creating a new blockchain node to deploy the
  blockchain validator(s) to.

  Fields:
    ethereumNodeDetails: Additional configuration specific to Ethereum
      blockchain nodes.
  """
    ethereumNodeDetails = _messages.MessageField('EthereumNodeDetails', 1)