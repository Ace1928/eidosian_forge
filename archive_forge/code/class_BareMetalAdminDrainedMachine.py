from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalAdminDrainedMachine(_messages.Message):
    """BareMetalAdminDrainedMachine represents the machines that are drained.

  Fields:
    nodeIp: Drained machine IP address.
  """
    nodeIp = _messages.StringField(1)