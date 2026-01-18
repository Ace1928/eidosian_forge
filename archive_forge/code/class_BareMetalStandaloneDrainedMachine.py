from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalStandaloneDrainedMachine(_messages.Message):
    """Represents a machine that is currently drained.

  Fields:
    nodeIp: Drained machine IP address.
  """
    nodeIp = _messages.StringField(1)