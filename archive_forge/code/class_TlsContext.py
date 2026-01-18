from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TlsContext(_messages.Message):
    """[Deprecated] The TLS settings for the client or server. The TLS settings
  for the client or server.

  Fields:
    certificateContext: Defines the mechanism to obtain the client or server
      certificate.
    validationContext: Defines the mechanism to obtain the Certificate
      Authority certificate to validate the client/server certificate. If
      omitted, the proxy will not validate the server or client certificate.
  """
    certificateContext = _messages.MessageField('TlsCertificateContext', 1)
    validationContext = _messages.MessageField('TlsValidationContext', 2)