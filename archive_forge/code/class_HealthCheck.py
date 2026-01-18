from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthCheck(_messages.Message):
    """Health checking configuration for VM instances. Unhealthy instances are
  killed and replaced with new instances. Only applicable for instances in App
  Engine flexible environment.

  Fields:
    checkInterval: Interval between health checks.
    disableHealthCheck: Whether to explicitly disable health checks for this
      instance.
    healthyThreshold: Number of consecutive successful health checks required
      before receiving traffic.
    host: Host header to send when performing an HTTP health check. Example:
      "myapp.appspot.com"
    restartThreshold: Number of consecutive failed health checks required
      before an instance is restarted.
    timeout: Time before the health check is considered failed.
    unhealthyThreshold: Number of consecutive failed health checks required
      before removing traffic.
  """
    checkInterval = _messages.StringField(1)
    disableHealthCheck = _messages.BooleanField(2)
    healthyThreshold = _messages.IntegerField(3, variant=_messages.Variant.UINT32)
    host = _messages.StringField(4)
    restartThreshold = _messages.IntegerField(5, variant=_messages.Variant.UINT32)
    timeout = _messages.StringField(6)
    unhealthyThreshold = _messages.IntegerField(7, variant=_messages.Variant.UINT32)