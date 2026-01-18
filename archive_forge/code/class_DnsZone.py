from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DnsZone(_messages.Message):
    """Represents a DNS zone resource.

  Fields:
    dnsSuffix: The DNS name suffix of this zone e.g. `example.com.`. Cloud DNS
      requires that a DNS suffix ends with a trailing dot.
    name: User assigned name for this resource. Must be unique within the
      project. The name must be 1-63 characters long, must begin with a
      letter, end with a letter or digit, and only contain lowercase letters,
      digits or dashes.
  """
    dnsSuffix = _messages.StringField(1)
    name = _messages.StringField(2)