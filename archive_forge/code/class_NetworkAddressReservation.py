from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkAddressReservation(_messages.Message):
    """A reservation of one or more addresses in a network.

  Fields:
    endAddress: The last address of this reservation block, inclusive. I.e.,
      for cases when reservations are only single addresses, end_address and
      start_address will be the same. Must be specified as a single IPv4
      address, e.g. 10.1.2.2.
    note: A note about this reservation, intended for human consumption.
    startAddress: The first address of this reservation block. Must be
      specified as a single IPv4 address, e.g. 10.1.2.2.
  """
    endAddress = _messages.StringField(1)
    note = _messages.StringField(2)
    startAddress = _messages.StringField(3)