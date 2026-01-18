from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SslCertsListResponse(_messages.Message):
    """SslCerts list response.

  Fields:
    items: List of client certificates for the instance.
    kind: This is always `sql#sslCertsList`.
  """
    items = _messages.MessageField('SslCert', 1, repeated=True)
    kind = _messages.StringField(2)