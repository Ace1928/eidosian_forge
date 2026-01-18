from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpFaultInjection(_messages.Message):
    """The specification for fault injection introduced into traffic to test
  the resiliency of clients to backend service failure. As part of fault
  injection, when clients send requests to a backend service, delays can be
  introduced by the load balancer on a percentage of requests before sending
  those request to the backend service. Similarly requests from clients can be
  aborted by the load balancer for a percentage of requests.

  Fields:
    abort: The specification for how client requests are aborted as part of
      fault injection.
    delay: The specification for how client requests are delayed as part of
      fault injection, before being sent to a backend service.
  """
    abort = _messages.MessageField('HttpFaultAbort', 1)
    delay = _messages.MessageField('HttpFaultDelay', 2)