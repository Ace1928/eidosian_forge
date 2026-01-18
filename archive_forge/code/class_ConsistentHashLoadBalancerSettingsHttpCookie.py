from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConsistentHashLoadBalancerSettingsHttpCookie(_messages.Message):
    """The information about the HTTP Cookie on which the hash function is
  based for load balancing policies that use a consistent hash.

  Fields:
    name: Name of the cookie.
    path: Path to set for the cookie.
    ttl: Lifetime of the cookie.
  """
    name = _messages.StringField(1)
    path = _messages.StringField(2)
    ttl = _messages.MessageField('Duration', 3)