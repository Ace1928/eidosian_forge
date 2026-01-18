from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpnGatewaysGetStatusResponse(_messages.Message):
    """A VpnGatewaysGetStatusResponse object.

  Fields:
    result: A VpnGatewayStatus attribute.
  """
    result = _messages.MessageField('VpnGatewayStatus', 1)