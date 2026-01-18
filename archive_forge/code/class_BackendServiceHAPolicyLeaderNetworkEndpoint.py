from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendServiceHAPolicyLeaderNetworkEndpoint(_messages.Message):
    """A BackendServiceHAPolicyLeaderNetworkEndpoint object.

  Fields:
    instance: Specifying the instance name of a leader is not supported.
  """
    instance = _messages.StringField(1)