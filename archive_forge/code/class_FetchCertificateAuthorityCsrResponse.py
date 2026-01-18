from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FetchCertificateAuthorityCsrResponse(_messages.Message):
    """Response message for
  CertificateAuthorityService.FetchCertificateAuthorityCsr.

  Fields:
    pemCsr: Output only. The PEM-encoded signed certificate signing request
      (CSR).
  """
    pemCsr = _messages.StringField(1)