from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpFaultDelay(_messages.Message):
    """Specifies the delay introduced by the load balancer before forwarding
  the request to the backend service as part of fault injection.

  Fields:
    fixedDelay: Specifies the value of the fixed delay interval.
    percentage: The percentage of traffic for connections, operations, or
      requests for which a delay is introduced as part of fault injection. The
      value must be from 0.0 to 100.0 inclusive.
  """
    fixedDelay = _messages.MessageField('Duration', 1)
    percentage = _messages.FloatField(2)