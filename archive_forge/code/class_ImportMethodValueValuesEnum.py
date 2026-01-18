from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportMethodValueValuesEnum(_messages.Enum):
    """Required. Immutable. The wrapping method to be used for incoming key
    material.

    Values:
      IMPORT_METHOD_UNSPECIFIED: Not specified.
      RSA_OAEP_3072_SHA1_AES_256: This ImportMethod represents the
        CKM_RSA_AES_KEY_WRAP key wrapping scheme defined in the PKCS #11
        standard. In summary, this involves wrapping the raw key with an
        ephemeral AES key, and wrapping the ephemeral AES key with a 3072 bit
        RSA key. For more details, see [RSA AES key wrap
        mechanism](http://docs.oasis-open.org/pkcs11/pkcs11-
        curr/v2.40/cos01/pkcs11-curr-v2.40-cos01.html#_Toc408226908).
      RSA_OAEP_4096_SHA1_AES_256: This ImportMethod represents the
        CKM_RSA_AES_KEY_WRAP key wrapping scheme defined in the PKCS #11
        standard. In summary, this involves wrapping the raw key with an
        ephemeral AES key, and wrapping the ephemeral AES key with a 4096 bit
        RSA key. For more details, see [RSA AES key wrap
        mechanism](http://docs.oasis-open.org/pkcs11/pkcs11-
        curr/v2.40/cos01/pkcs11-curr-v2.40-cos01.html#_Toc408226908).
      RSA_OAEP_3072_SHA256_AES_256: This ImportMethod represents the
        CKM_RSA_AES_KEY_WRAP key wrapping scheme defined in the PKCS #11
        standard. In summary, this involves wrapping the raw key with an
        ephemeral AES key, and wrapping the ephemeral AES key with a 3072 bit
        RSA key. For more details, see [RSA AES key wrap
        mechanism](http://docs.oasis-open.org/pkcs11/pkcs11-
        curr/v2.40/cos01/pkcs11-curr-v2.40-cos01.html#_Toc408226908).
      RSA_OAEP_4096_SHA256_AES_256: This ImportMethod represents the
        CKM_RSA_AES_KEY_WRAP key wrapping scheme defined in the PKCS #11
        standard. In summary, this involves wrapping the raw key with an
        ephemeral AES key, and wrapping the ephemeral AES key with a 4096 bit
        RSA key. For more details, see [RSA AES key wrap
        mechanism](http://docs.oasis-open.org/pkcs11/pkcs11-
        curr/v2.40/cos01/pkcs11-curr-v2.40-cos01.html#_Toc408226908).
      RSA_OAEP_3072_SHA256: This ImportMethod represents RSAES-OAEP with a
        3072 bit RSA key. The key material to be imported is wrapped directly
        with the RSA key. Due to technical limitations of RSA wrapping, this
        method cannot be used to wrap RSA keys for import.
      RSA_OAEP_4096_SHA256: This ImportMethod represents RSAES-OAEP with a
        4096 bit RSA key. The key material to be imported is wrapped directly
        with the RSA key. Due to technical limitations of RSA wrapping, this
        method cannot be used to wrap RSA keys for import.
    """
    IMPORT_METHOD_UNSPECIFIED = 0
    RSA_OAEP_3072_SHA1_AES_256 = 1
    RSA_OAEP_4096_SHA1_AES_256 = 2
    RSA_OAEP_3072_SHA256_AES_256 = 3
    RSA_OAEP_4096_SHA256_AES_256 = 4
    RSA_OAEP_3072_SHA256 = 5
    RSA_OAEP_4096_SHA256 = 6