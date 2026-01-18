from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnrollVmwareNodePoolRequest(_messages.Message):
    """Message for enrolling a VMware node pool.

  Fields:
    vmwareNodePoolId: The target node pool id to be enrolled.
  """
    vmwareNodePoolId = _messages.StringField(1)