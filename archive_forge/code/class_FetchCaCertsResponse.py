from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FetchCaCertsResponse(_messages.Message):
    """Response message for CertificateAuthorityService.FetchCaCerts.

  Fields:
    caCerts: The PEM encoded CA certificate chains of all certificate
      authorities in this CaPool in the ENABLED, DISABLED, or STAGED states.
  """
    caCerts = _messages.MessageField('CertChain', 1, repeated=True)