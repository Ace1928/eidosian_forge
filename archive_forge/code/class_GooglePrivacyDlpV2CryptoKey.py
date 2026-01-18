from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CryptoKey(_messages.Message):
    """This is a data encryption key (DEK) (as opposed to a key encryption key
  (KEK) stored by Cloud Key Management Service (Cloud KMS). When using Cloud
  KMS to wrap or unwrap a DEK, be sure to set an appropriate IAM policy on the
  KEK to ensure an attacker cannot unwrap the DEK.

  Fields:
    kmsWrapped: Key wrapped using Cloud KMS
    transient: Transient crypto key
    unwrapped: Unwrapped crypto key
  """
    kmsWrapped = _messages.MessageField('GooglePrivacyDlpV2KmsWrappedCryptoKey', 1)
    transient = _messages.MessageField('GooglePrivacyDlpV2TransientCryptoKey', 2)
    unwrapped = _messages.MessageField('GooglePrivacyDlpV2UnwrappedCryptoKey', 3)