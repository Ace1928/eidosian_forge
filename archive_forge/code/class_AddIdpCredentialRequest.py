from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AddIdpCredentialRequest(_messages.Message):
    """The request for creating an IdpCredential with its associated payload.
  An InboundSamlSsoProfile can own up to 2 credentials.

  Fields:
    pemData: PEM encoded x509 certificate containing the public key for
      verifying IdP signatures.
  """
    pemData = _messages.StringField(1)