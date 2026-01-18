from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NatIpInfoResponse(_messages.Message):
    """A NatIpInfoResponse object.

  Fields:
    result: [Output Only] A list of NAT IP information.
  """
    result = _messages.MessageField('NatIpInfo', 1, repeated=True)