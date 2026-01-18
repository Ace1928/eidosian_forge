from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowedIpRange(_messages.Message):
    """Allowed IP range with user-provided description.

  Fields:
    description: Optional. User-provided description. It must contain at most
      300 characters.
    value: IP address or range, defined using CIDR notation, of requests that
      this rule applies to. Examples: `192.168.1.1` or `192.168.0.0/16` or
      `2001:db8::/32` or `2001:0db8:0000:0042:0000:8a2e:0370:7334`. IP range
      prefixes should be properly truncated. For example, `1.2.3.4/24` should
      be truncated to `1.2.3.0/24`. Similarly, for IPv6, `2001:db8::1/32`
      should be truncated to `2001:db8::/32`.
  """
    description = _messages.StringField(1)
    value = _messages.StringField(2)