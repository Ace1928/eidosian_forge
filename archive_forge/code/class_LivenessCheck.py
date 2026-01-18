from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LivenessCheck(_messages.Message):
    """Health checking configuration for VM instances. Unhealthy instances are
  killed and replaced with new instances.

  Fields:
    checkInterval: Interval between health checks.
    failureThreshold: Number of consecutive failed checks required before
      considering the VM unhealthy.
    host: Host header to send when performing a HTTP Liveness check. Example:
      "myapp.appspot.com"
    initialDelay: The initial delay before starting to execute the checks.
    path: The request path.
    successThreshold: Number of consecutive successful checks required before
      considering the VM healthy.
    timeout: Time before the check is considered failed.
  """
    checkInterval = _messages.StringField(1)
    failureThreshold = _messages.IntegerField(2, variant=_messages.Variant.UINT32)
    host = _messages.StringField(3)
    initialDelay = _messages.StringField(4)
    path = _messages.StringField(5)
    successThreshold = _messages.IntegerField(6, variant=_messages.Variant.UINT32)
    timeout = _messages.StringField(7)