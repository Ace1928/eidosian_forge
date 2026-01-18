from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReadinessCheck(_messages.Message):
    """Readiness checking configuration for VM instances. Unhealthy instances
  are removed from traffic rotation.

  Fields:
    appStartTimeout: A maximum time limit on application initialization,
      measured from moment the application successfully replies to a
      healthcheck until it is ready to serve traffic.
    checkInterval: Interval between health checks.
    failureThreshold: Number of consecutive failed checks required before
      removing traffic.
    host: Host header to send when performing a HTTP Readiness check. Example:
      "myapp.appspot.com"
    path: The request path.
    successThreshold: Number of consecutive successful checks required before
      receiving traffic.
    timeout: Time before the check is considered failed.
  """
    appStartTimeout = _messages.StringField(1)
    checkInterval = _messages.StringField(2)
    failureThreshold = _messages.IntegerField(3, variant=_messages.Variant.UINT32)
    host = _messages.StringField(4)
    path = _messages.StringField(5)
    successThreshold = _messages.IntegerField(6, variant=_messages.Variant.UINT32)
    timeout = _messages.StringField(7)