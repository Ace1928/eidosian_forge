from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectAttachmentPrivateInfo(_messages.Message):
    """Information for an interconnect attachment when this belongs to an
  interconnect of type DEDICATED.

  Fields:
    tag8021q: [Output Only] 802.1q encapsulation tag to be used for traffic
      between Google and the customer, going to and from this network and
      region.
  """
    tag8021q = _messages.IntegerField(1, variant=_messages.Variant.UINT32)