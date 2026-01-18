from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectionDraining(_messages.Message):
    """Message containing connection draining configuration.

  Fields:
    drainingTimeoutSec: Configures a duration timeout for existing requests on
      a removed backend instance. For supported load balancers and protocols,
      as described in Enabling connection draining.
  """
    drainingTimeoutSec = _messages.IntegerField(1, variant=_messages.Variant.INT32)