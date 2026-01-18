from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AddressSocketAddress(_messages.Message):
    """Specifies an IP:Port address.

  Fields:
    address: Required. Specifies an IPV4 address. CIDR are not allowed.
    port: Required. Specifies the port.
  """
    address = _messages.StringField(1)
    port = _messages.IntegerField(2, variant=_messages.Variant.INT32)