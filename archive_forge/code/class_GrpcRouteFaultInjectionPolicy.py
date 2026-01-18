from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrpcRouteFaultInjectionPolicy(_messages.Message):
    """The specification for fault injection introduced into traffic to test
  the resiliency of clients to destination service failure. As part of fault
  injection, when clients send requests to a destination, delays can be
  introduced on a percentage of requests before sending those requests to the
  destination service. Similarly requests from clients can be aborted by for a
  percentage of requests.

  Fields:
    abort: The specification for aborting to client requests.
    delay: The specification for injecting delay to client requests.
  """
    abort = _messages.MessageField('GrpcRouteFaultInjectionPolicyAbort', 1)
    delay = _messages.MessageField('GrpcRouteFaultInjectionPolicyDelay', 2)