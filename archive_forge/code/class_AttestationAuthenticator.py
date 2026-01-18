from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttestationAuthenticator(_messages.Message):
    """An attestation authenticator that will be used to verify attestations.
  Typically this is just a set of public keys. Conceptually, an authenticator
  can be treated as always returning either "authenticated" or "not
  authenticated" when presented with a signed attestation (almost always
  assumed to be a [DSSE](https://github.com/secure-systems-lab/dsse)
  attestation). The details of how an authenticator makes this decision are
  specific to the type of 'authenticator' that this message wraps.

  Fields:
    displayName: Optional. A user-provided name for this
      `AttestationAuthenticator`. This field has no effect on the policy
      evaluation behavior except to improve readability of messages in
      evaluation results.
    pkixPublicKeySet: Optional. A set of raw PKIX SubjectPublicKeyInfo format
      public keys. If any public key in the set validates the attestation
      signature, then the signature is considered authenticated (i.e. any one
      key is sufficient to authenticate).
  """
    displayName = _messages.StringField(1)
    pkixPublicKeySet = _messages.MessageField('PkixPublicKeySet', 2)