from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1PrivatePasswordLeakVerification(_messages.Message):
    """Private password leak verification info.

  Fields:
    encryptedLeakMatchPrefixes: Output only. List of prefixes of the encrypted
      potential password leaks that matched the given parameters. They must be
      compared with the client-side decryption prefix of
      `reencrypted_user_credentials_hash`
    encryptedUserCredentialsHash: Optional. Encrypted Scrypt hash of the
      canonicalized username+password. It is re-encrypted by the server and
      returned through `reencrypted_user_credentials_hash`.
    lookupHashPrefix: Required. Exactly 26-bit prefix of the SHA-256 hash of
      the canonicalized username. It is used to look up password leaks
      associated with that hash prefix.
    reencryptedUserCredentialsHash: Output only. Corresponds to the re-
      encryption of the `encrypted_user_credentials_hash` field. It is used to
      match potential password leaks within `encrypted_leak_match_prefixes`.
  """
    encryptedLeakMatchPrefixes = _messages.BytesField(1, repeated=True)
    encryptedUserCredentialsHash = _messages.BytesField(2)
    lookupHashPrefix = _messages.BytesField(3)
    reencryptedUserCredentialsHash = _messages.BytesField(4)