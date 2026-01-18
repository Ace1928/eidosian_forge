from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IntermediateCA(_messages.Message):
    """Intermediate CA certificates used for building the trust chain to trust
  anchor

  Fields:
    pemCertificate: PEM certificate of the PKI used for validation. Must only
      contain one ca certificate.
  """
    pemCertificate = _messages.StringField(1)