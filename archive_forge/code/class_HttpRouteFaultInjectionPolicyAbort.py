from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpRouteFaultInjectionPolicyAbort(_messages.Message):
    """Specification of how client requests are aborted as part of fault
  injection before being sent to a destination.

  Fields:
    httpStatus: The HTTP status code used to abort the request. The value must
      be between 200 and 599 inclusive.
    percentage: The percentage of traffic which will be aborted. The value
      must be between [0, 100]
  """
    httpStatus = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    percentage = _messages.IntegerField(2, variant=_messages.Variant.INT32)