from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerStatusVersionTarget(_messages.Message):
    """A InstanceGroupManagerStatusVersionTarget object.

  Fields:
    isReached: [Output Only] A bit indicating whether version target has been
      reached in this managed instance group, i.e. all instances are in their
      target version. Instances' target version are specified by version field
      on Instance Group Manager.
  """
    isReached = _messages.BooleanField(1)