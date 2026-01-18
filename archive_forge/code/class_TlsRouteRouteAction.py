from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TlsRouteRouteAction(_messages.Message):
    """The specifications for routing traffic and applying associated policies.

  Fields:
    destinations: Required. The destination services to which traffic should
      be forwarded. At least one destination service is required.
    idleTimeout: Optional. Specifies the idle timeout for the selected route.
      The idle timeout is defined as the period in which there are no bytes
      sent or received on either the upstream or downstream connection. If not
      set, the default idle timeout is 1 hour. If set to 0s, the timeout will
      be disabled.
  """
    destinations = _messages.MessageField('TlsRouteRouteDestination', 1, repeated=True)
    idleTimeout = _messages.StringField(2)