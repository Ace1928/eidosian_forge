from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectDiagnosticsMacsecStatus(_messages.Message):
    """Describes the status of MACsec encryption on the link.

  Fields:
    ckn: Indicates the Connectivity Association Key Name (CKN) currently being
      used if MACsec is operational.
    operational: Indicates whether or not MACsec is operational on this link.
  """
    ckn = _messages.StringField(1)
    operational = _messages.BooleanField(2)