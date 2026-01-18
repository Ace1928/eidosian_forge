from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DnsResourceRecord(_messages.Message):
    """The structure describing the DNS Resource Record that needs to be added
  to DNS configuration for the authorization to be usable by certificate.

  Fields:
    data: Output only. Data of the DNS Resource Record.
    name: Output only. Fully qualified name of the DNS Resource Record. e.g.
      `_acme-challenge.example.com`
    type: Output only. Type of the DNS Resource Record. Currently always set
      to "CNAME".
  """
    data = _messages.StringField(1)
    name = _messages.StringField(2)
    type = _messages.StringField(3)