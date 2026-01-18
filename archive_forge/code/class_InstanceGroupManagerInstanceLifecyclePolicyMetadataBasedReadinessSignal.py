from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerInstanceLifecyclePolicyMetadataBasedReadinessSignal(_messages.Message):
    """A
  InstanceGroupManagerInstanceLifecyclePolicyMetadataBasedReadinessSignal
  object.

  Fields:
    timeoutSec: The number of seconds to wait for a readiness signal during
      initialization before timing out.
  """
    timeoutSec = _messages.IntegerField(1, variant=_messages.Variant.INT32)