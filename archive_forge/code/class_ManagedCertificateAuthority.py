from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedCertificateAuthority(_messages.Message):
    """A ManagedCertificateAuthority object.

  Fields:
    caCerts: The PEM encoded CA certificate chains for redis managed server
      authentication
  """
    caCerts = _messages.MessageField('CertChain', 1, repeated=True)