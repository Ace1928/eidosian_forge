from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Thresholds(_messages.Message):
    """Thresholds define the utilization of resources triggering scale-out and
  scale-in operations.

  Fields:
    scaleIn: Required. The utilization triggering the scale-in operation in
      percent.
    scaleOut: Required. The utilization triggering the scale-out operation in
      percent.
  """
    scaleIn = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    scaleOut = _messages.IntegerField(2, variant=_messages.Variant.INT32)