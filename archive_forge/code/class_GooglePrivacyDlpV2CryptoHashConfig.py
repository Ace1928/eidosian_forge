from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CryptoHashConfig(_messages.Message):
    """Pseudonymization method that generates surrogates via cryptographic
  hashing. Uses SHA-256. The key size must be either 32 or 64 bytes. Outputs a
  base64 encoded representation of the hashed output (for example,
  L7k0BHmF1ha5U3NfGykjro4xWi1MPVQPjhMAZbSV9mM=). Currently, only string and
  integer values can be hashed. See https://cloud.google.com/sensitive-data-
  protection/docs/pseudonymization to learn more.

  Fields:
    cryptoKey: The key used by the hash function.
  """
    cryptoKey = _messages.MessageField('GooglePrivacyDlpV2CryptoKey', 1)