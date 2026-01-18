from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SignedUrlKey(_messages.Message):
    """Represents a customer-supplied Signing Key used by Cloud CDN Signed URLs

  Fields:
    keyName: Name of the key. The name must be 1-63 characters long, and
      comply with RFC1035. Specifically, the name must be 1-63 characters long
      and match the regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which
      means the first character must be a lowercase letter, and all following
      characters must be a dash, lowercase letter, or digit, except the last
      character, which cannot be a dash.
    keyValue: 128-bit key value used for signing the URL. The key value must
      be a valid RFC 4648 Section 5 base64url encoded string.
  """
    keyName = _messages.StringField(1)
    keyValue = _messages.StringField(2)