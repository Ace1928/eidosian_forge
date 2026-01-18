from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivateConnectivity(_messages.Message):
    """Private Connectivity.

  Fields:
    privateConnection: Required. The resource name (URI) of the private
      connection.
  """
    privateConnection = _messages.StringField(1)