from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RenameNetworkRequest(_messages.Message):
    """Message requesting rename of a server.

  Fields:
    newNetworkId: Required. The new `id` of the network.
  """
    newNetworkId = _messages.StringField(1)