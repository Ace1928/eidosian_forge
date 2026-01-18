from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AssignmentOsType(_messages.Message):
    """Defines the criteria for selecting VM Instances by OS type.

  Fields:
    osArchitecture: Targets VM instances with OS Inventory enabled and having
      the following OS architecture.
    osShortName: Targets VM instances with OS Inventory enabled and having the
      following OS short name, for example "debian" or "windows".
    osVersion: Targets VM instances with OS Inventory enabled and having the
      following following OS version.
  """
    osArchitecture = _messages.StringField(1)
    osShortName = _messages.StringField(2)
    osVersion = _messages.StringField(3)