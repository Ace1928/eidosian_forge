from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FaultInjectionPolicy(_messages.Message):
    """The specification for fault injection introduced into traffic to test
  the resiliency of clients to destination service failure. As part of fault
  injection, when clients send requests to a destination, delays can be
  introduced by client proxy on a percentage of requests before sending those
  requests to the destination service. Similarly requests can be aborted by
  client proxy for a percentage of requests.

  Fields:
    abort: Abort condtion
    delay: fixed delay time
  """
    abort = _messages.MessageField('Abort', 1)
    delay = _messages.MessageField('Delay', 2)