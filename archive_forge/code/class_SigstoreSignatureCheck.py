from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SigstoreSignatureCheck(_messages.Message):
    """A Sigstore signature check, which verifies the Sigstore signature
  associated with an image.

  Fields:
    sigstoreAuthorities: Required. The authorities required by this check to
      verify the signature. A signature only needs to be verified by one
      authority to pass the check.
  """
    sigstoreAuthorities = _messages.MessageField('SigstoreAuthority', 1, repeated=True)