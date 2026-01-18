from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EncryptionConfig(_messages.Message):
    """Encryption configuration for a Cloud Spanner database.

  Fields:
    kmsKeyName: The Cloud KMS key to be used for encrypting and decrypting the
      database. Values are of the form
      `projects//locations//keyRings//cryptoKeys/`.
    kmsKeyNames: Specifies the KMS configuration for the one or more keys used
      to encrypt the database. Values are of the form
      `projects//locations//keyRings//cryptoKeys/`. The keys referenced by
      kms_key_names must fully cover all regions of the database instance
      configuration. Some examples: * For single region database instance
      configs, specify a single regional location KMS key. * For multi-
      regional database instance configs of type GOOGLE_MANAGED, either
      specify a multi-regional location KMS key or multiple regional location
      KMS keys that cover all regions in the instance config. * For a database
      instance config of type USER_MANAGED, please specify only regional
      location KMS keys to cover each region in the instance config. Multi-
      regional location KMS keys are not supported for USER_MANAGED instance
      configs.
  """
    kmsKeyName = _messages.StringField(1)
    kmsKeyNames = _messages.StringField(2, repeated=True)