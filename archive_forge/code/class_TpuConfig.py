from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TpuConfig(_messages.Message):
    """Configuration for Cloud TPU.

  Fields:
    enabled: Whether Cloud TPU integration is enabled or not.
    ipv4CidrBlock: IPv4 CIDR block reserved for Cloud TPU in the VPC.
    useServiceNetworking: Whether to use service networking for Cloud TPU or
      not.
  """
    enabled = _messages.BooleanField(1)
    ipv4CidrBlock = _messages.StringField(2)
    useServiceNetworking = _messages.BooleanField(3)