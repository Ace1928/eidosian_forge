from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectCircuitInfo(_messages.Message):
    """Describes a single physical circuit between the Customer and Google.
  CircuitInfo objects are created by Google, so all fields are output only.

  Fields:
    customerDemarcId: Customer-side demarc ID for this circuit.
    googleCircuitId: Google-assigned unique ID for this circuit. Assigned at
      circuit turn-up.
    googleDemarcId: Google-side demarc ID for this circuit. Assigned at
      circuit turn-up and provided by Google to the customer in the LOA.
  """
    customerDemarcId = _messages.StringField(1)
    googleCircuitId = _messages.StringField(2)
    googleDemarcId = _messages.StringField(3)