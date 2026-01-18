from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalAdminDrainingMachine(_messages.Message):
    """BareMetalAdminDrainingMachine represents the machines that are currently
  draining.

  Fields:
    nodeIp: Draining machine IP address.
    podCount: The count of pods yet to drain.
  """
    nodeIp = _messages.StringField(1)
    podCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)