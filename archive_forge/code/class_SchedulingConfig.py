from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchedulingConfig(_messages.Message):
    """Sets the scheduling options for this node.

  Fields:
    preemptible: Defines whether the node is preemptible.
    reserved: Whether the node is created under a reservation.
    spot: Optional. Defines whether the node is Spot VM.
  """
    preemptible = _messages.BooleanField(1)
    reserved = _messages.BooleanField(2)
    spot = _messages.BooleanField(3)