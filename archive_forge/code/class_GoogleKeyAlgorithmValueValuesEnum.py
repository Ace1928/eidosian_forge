from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleKeyAlgorithmValueValuesEnum(_messages.Enum):
    """The hashing algorithm used for signature verification. It will only be
    present in the case of Google managed keys.

    Values:
      CRYPTO_KEY_VERSION_ALGORITHM_UNSPECIFIED: Not specified.
      GOOGLE_SYMMETRIC_ENCRYPTION: Creates symmetric encryption keys.
      AES_128_GCM: AES-GCM (Galois Counter Mode) using 128-bit keys.
      AES_256_GCM: AES-GCM (Galois Counter Mode) using 256-bit keys.
      AES_128_CBC: AES-CBC (Cipher Block Chaining Mode) using 128-bit keys.
      AES_256_CBC: AES-CBC (Cipher Block Chaining Mode) using 256-bit keys.
      AES_128_CTR: AES-CTR (Counter Mode) using 128-bit keys.
      AES_256_CTR: AES-CTR (Counter Mode) using 256-bit keys.
      RSA_SIGN_PSS_2048_SHA256: RSASSA-PSS 2048 bit key with a SHA256 digest.
      RSA_SIGN_PSS_3072_SHA256: RSASSA-PSS 3072 bit key with a SHA256 digest.
      RSA_SIGN_PSS_4096_SHA256: RSASSA-PSS 4096 bit key with a SHA256 digest.
      RSA_SIGN_PSS_4096_SHA512: RSASSA-PSS 4096 bit key with a SHA512 digest.
      RSA_SIGN_PKCS1_2048_SHA256: RSASSA-PKCS1-v1_5 with a 2048 bit key and a
        SHA256 digest.
      RSA_SIGN_PKCS1_3072_SHA256: RSASSA-PKCS1-v1_5 with a 3072 bit key and a
        SHA256 digest.
      RSA_SIGN_PKCS1_4096_SHA256: RSASSA-PKCS1-v1_5 with a 4096 bit key and a
        SHA256 digest.
      RSA_SIGN_PKCS1_4096_SHA512: RSASSA-PKCS1-v1_5 with a 4096 bit key and a
        SHA512 digest.
      RSA_SIGN_RAW_PKCS1_2048: RSASSA-PKCS1-v1_5 signing without encoding,
        with a 2048 bit key.
      RSA_SIGN_RAW_PKCS1_3072: RSASSA-PKCS1-v1_5 signing without encoding,
        with a 3072 bit key.
      RSA_SIGN_RAW_PKCS1_4096: RSASSA-PKCS1-v1_5 signing without encoding,
        with a 4096 bit key.
      RSA_DECRYPT_OAEP_2048_SHA256: RSAES-OAEP 2048 bit key with a SHA256
        digest.
      RSA_DECRYPT_OAEP_3072_SHA256: RSAES-OAEP 3072 bit key with a SHA256
        digest.
      RSA_DECRYPT_OAEP_4096_SHA256: RSAES-OAEP 4096 bit key with a SHA256
        digest.
      RSA_DECRYPT_OAEP_4096_SHA512: RSAES-OAEP 4096 bit key with a SHA512
        digest.
      RSA_DECRYPT_OAEP_2048_SHA1: RSAES-OAEP 2048 bit key with a SHA1 digest.
      RSA_DECRYPT_OAEP_3072_SHA1: RSAES-OAEP 3072 bit key with a SHA1 digest.
      RSA_DECRYPT_OAEP_4096_SHA1: RSAES-OAEP 4096 bit key with a SHA1 digest.
      EC_SIGN_P256_SHA256: ECDSA on the NIST P-256 curve with a SHA256 digest.
        Other hash functions can also be used:
        https://cloud.google.com/kms/docs/create-validate-
        signatures#ecdsa_support_for_other_hash_algorithms
      EC_SIGN_P384_SHA384: ECDSA on the NIST P-384 curve with a SHA384 digest.
        Other hash functions can also be used:
        https://cloud.google.com/kms/docs/create-validate-
        signatures#ecdsa_support_for_other_hash_algorithms
      EC_SIGN_SECP256K1_SHA256: ECDSA on the non-NIST secp256k1 curve. This
        curve is only supported for HSM protection level. Other hash functions
        can also be used: https://cloud.google.com/kms/docs/create-validate-
        signatures#ecdsa_support_for_other_hash_algorithms
      HMAC_SHA256: HMAC-SHA256 signing with a 256 bit key.
      HMAC_SHA1: HMAC-SHA1 signing with a 160 bit key.
      HMAC_SHA384: HMAC-SHA384 signing with a 384 bit key.
      HMAC_SHA512: HMAC-SHA512 signing with a 512 bit key.
      HMAC_SHA224: HMAC-SHA224 signing with a 224 bit key.
      EXTERNAL_SYMMETRIC_ENCRYPTION: Algorithm representing symmetric
        encryption by an external key manager.
    """
    CRYPTO_KEY_VERSION_ALGORITHM_UNSPECIFIED = 0
    GOOGLE_SYMMETRIC_ENCRYPTION = 1
    AES_128_GCM = 2
    AES_256_GCM = 3
    AES_128_CBC = 4
    AES_256_CBC = 5
    AES_128_CTR = 6
    AES_256_CTR = 7
    RSA_SIGN_PSS_2048_SHA256 = 8
    RSA_SIGN_PSS_3072_SHA256 = 9
    RSA_SIGN_PSS_4096_SHA256 = 10
    RSA_SIGN_PSS_4096_SHA512 = 11
    RSA_SIGN_PKCS1_2048_SHA256 = 12
    RSA_SIGN_PKCS1_3072_SHA256 = 13
    RSA_SIGN_PKCS1_4096_SHA256 = 14
    RSA_SIGN_PKCS1_4096_SHA512 = 15
    RSA_SIGN_RAW_PKCS1_2048 = 16
    RSA_SIGN_RAW_PKCS1_3072 = 17
    RSA_SIGN_RAW_PKCS1_4096 = 18
    RSA_DECRYPT_OAEP_2048_SHA256 = 19
    RSA_DECRYPT_OAEP_3072_SHA256 = 20
    RSA_DECRYPT_OAEP_4096_SHA256 = 21
    RSA_DECRYPT_OAEP_4096_SHA512 = 22
    RSA_DECRYPT_OAEP_2048_SHA1 = 23
    RSA_DECRYPT_OAEP_3072_SHA1 = 24
    RSA_DECRYPT_OAEP_4096_SHA1 = 25
    EC_SIGN_P256_SHA256 = 26
    EC_SIGN_P384_SHA384 = 27
    EC_SIGN_SECP256K1_SHA256 = 28
    HMAC_SHA256 = 29
    HMAC_SHA1 = 30
    HMAC_SHA384 = 31
    HMAC_SHA512 = 32
    HMAC_SHA224 = 33
    EXTERNAL_SYMMETRIC_ENCRYPTION = 34