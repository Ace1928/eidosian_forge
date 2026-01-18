from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportJob(_messages.Message):
    """An ImportJob can be used to create CryptoKeys and CryptoKeyVersions
  using pre-existing key material, generated outside of Cloud KMS. When an
  ImportJob is created, Cloud KMS will generate a "wrapping key", which is a
  public/private key pair. You use the wrapping key to encrypt (also known as
  wrap) the pre-existing key material to protect it during the import process.
  The nature of the wrapping key depends on the choice of import_method. When
  the wrapping key generation is complete, the state will be set to ACTIVE and
  the public_key can be fetched. The fetched public key can then be used to
  wrap your pre-existing key material. Once the key material is wrapped, it
  can be imported into a new CryptoKeyVersion in an existing CryptoKey by
  calling ImportCryptoKeyVersion. Multiple CryptoKeyVersions can be imported
  with a single ImportJob. Cloud KMS uses the private key portion of the
  wrapping key to unwrap the key material. Only Cloud KMS has access to the
  private key. An ImportJob expires 3 days after it is created. Once expired,
  Cloud KMS will no longer be able to import or unwrap any key material that
  was wrapped with the ImportJob's public key. For more information, see
  [Importing a key](https://cloud.google.com/kms/docs/importing-a-key).

  Enums:
    ImportMethodValueValuesEnum: Required. Immutable. The wrapping method to
      be used for incoming key material.
    ProtectionLevelValueValuesEnum: Required. Immutable. The protection level
      of the ImportJob. This must match the protection_level of the
      version_template on the CryptoKey you attempt to import into.
    StateValueValuesEnum: Output only. The current state of the ImportJob,
      indicating if it can be used.

  Fields:
    attestation: Output only. Statement that was generated and signed by the
      key creator (for example, an HSM) at key creation time. Use this
      statement to verify attributes of the key as stored on the HSM,
      independently of Google. Only present if the chosen ImportMethod is one
      with a protection level of HSM.
    createTime: Output only. The time at which this ImportJob was created.
    expireEventTime: Output only. The time this ImportJob expired. Only
      present if state is EXPIRED.
    expireTime: Output only. The time at which this ImportJob is scheduled for
      expiration and can no longer be used to import key material.
    generateTime: Output only. The time this ImportJob's key material was
      generated.
    importMethod: Required. Immutable. The wrapping method to be used for
      incoming key material.
    name: Output only. The resource name for this ImportJob in the format
      `projects/*/locations/*/keyRings/*/importJobs/*`.
    protectionLevel: Required. Immutable. The protection level of the
      ImportJob. This must match the protection_level of the version_template
      on the CryptoKey you attempt to import into.
    publicKey: Output only. The public key with which to wrap key material
      prior to import. Only returned if state is ACTIVE.
    state: Output only. The current state of the ImportJob, indicating if it
      can be used.
  """

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

    class ProtectionLevelValueValuesEnum(_messages.Enum):
        """Required. Immutable. The protection level of the ImportJob. This must
    match the protection_level of the version_template on the CryptoKey you
    attempt to import into.

    Values:
      PROTECTION_LEVEL_UNSPECIFIED: Not specified.
      SOFTWARE: Crypto operations are performed in software.
      HSM: Crypto operations are performed in a Hardware Security Module.
      EXTERNAL: Crypto operations are performed by an external key manager.
      EXTERNAL_VPC: Crypto operations are performed in an EKM-over-VPC
        backend.
    """
        PROTECTION_LEVEL_UNSPECIFIED = 0
        SOFTWARE = 1
        HSM = 2
        EXTERNAL = 3
        EXTERNAL_VPC = 4

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the ImportJob, indicating if it can
    be used.

    Values:
      IMPORT_JOB_STATE_UNSPECIFIED: Not specified.
      PENDING_GENERATION: The wrapping key for this job is still being
        generated. It may not be used. Cloud KMS will automatically mark this
        job as ACTIVE as soon as the wrapping key is generated.
      ACTIVE: This job may be used in CreateCryptoKey and
        CreateCryptoKeyVersion requests.
      EXPIRED: This job can no longer be used and may not leave this state
        once entered.
    """
        IMPORT_JOB_STATE_UNSPECIFIED = 0
        PENDING_GENERATION = 1
        ACTIVE = 2
        EXPIRED = 3
    attestation = _messages.MessageField('KeyOperationAttestation', 1)
    createTime = _messages.StringField(2)
    expireEventTime = _messages.StringField(3)
    expireTime = _messages.StringField(4)
    generateTime = _messages.StringField(5)
    importMethod = _messages.EnumField('ImportMethodValueValuesEnum', 6)
    name = _messages.StringField(7)
    protectionLevel = _messages.EnumField('ProtectionLevelValueValuesEnum', 8)
    publicKey = _messages.MessageField('WrappingPublicKey', 9)
    state = _messages.EnumField('StateValueValuesEnum', 10)