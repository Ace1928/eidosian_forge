from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TrafficConfig(_messages.Message):
    """Network traffic configuration.

  Fields:
    anticipatedTrafficMatrix: Traffic Matrix for anticipated network traffic
  """
    anticipatedTrafficMatrix = _messages.MessageField('LogicalTrafficMatrix', 1)