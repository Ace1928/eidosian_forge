from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TrustAnchor(_messages.Message):
    """TrustAnchor is the root of trust of x509 federation.

  Fields:
    pemCertificate: PEM certificate of the PKI used for validation. Must only
      contain one ca certificate, and must be self-signed.
  """
    pemCertificate = _messages.StringField(1)