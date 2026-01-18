from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectMacsecConfigPreSharedKey(_messages.Message):
    """Describes a pre-shared key used to setup MACsec in static connectivity
  association key (CAK) mode.

  Fields:
    cak: An auto-generated Connectivity Association Key (CAK) for this key.
    ckn: An auto-generated Connectivity Association Key Name (CKN) for this
      key.
    name: User provided name for this pre-shared key.
    startTime: User provided timestamp on or after which this key is valid.
  """
    cak = _messages.StringField(1)
    ckn = _messages.StringField(2)
    name = _messages.StringField(3)
    startTime = _messages.StringField(4)