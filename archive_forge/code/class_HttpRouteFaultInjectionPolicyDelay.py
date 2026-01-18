from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpRouteFaultInjectionPolicyDelay(_messages.Message):
    """Specification of how client requests are delayed as part of fault
  injection before being sent to a destination.

  Fields:
    fixedDelay: Specify a fixed delay before forwarding the request.
    percentage: The percentage of traffic on which delay will be injected. The
      value must be between [0, 100]
  """
    fixedDelay = _messages.StringField(1)
    percentage = _messages.IntegerField(2, variant=_messages.Variant.INT32)