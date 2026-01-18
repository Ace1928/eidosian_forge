from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NatIpInfo(_messages.Message):
    """Contains NAT IP information of a NAT config (i.e. usage status, mode).

  Fields:
    natIpInfoMappings: A list of all NAT IPs assigned to this NAT config.
    natName: Name of the NAT config which the NAT IP belongs to.
  """
    natIpInfoMappings = _messages.MessageField('NatIpInfoNatIpInfoMapping', 1, repeated=True)
    natName = _messages.StringField(2)