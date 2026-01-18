from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenerateUploadUrlRequest(_messages.Message):
    """Request of `GenerateSourceUploadUrl` method.

  Enums:
    EnvironmentValueValuesEnum: The function environment the generated upload
      url will be used for. The upload url for 2nd Gen functions can also be
      used for 1st gen functions, but not vice versa. If not specified, 2nd
      generation-style upload URLs are generated.

  Fields:
    environment: The function environment the generated upload url will be
      used for. The upload url for 2nd Gen functions can also be used for 1st
      gen functions, but not vice versa. If not specified, 2nd generation-
      style upload URLs are generated.
    kmsKeyName: [Preview] Resource name of a KMS crypto key (managed by the
      user) used to encrypt/decrypt function source code objects in
      intermediate Cloud Storage buckets. When you generate an upload url and
      upload your source code, it gets copied to an intermediate Cloud Storage
      bucket. The source code is then copied to a versioned directory in the
      sources bucket in the consumer project during the function deployment.
      It must match the pattern `projects/{project}/locations/{location}/keyRi
      ngs/{key_ring}/cryptoKeys/{crypto_key}`. The Google Cloud Functions
      service account (service-{project_number}@gcf-admin-
      robot.iam.gserviceaccount.com) must be granted the role 'Cloud KMS
      CryptoKey Encrypter/Decrypter
      (roles/cloudkms.cryptoKeyEncrypterDecrypter)' on the
      Key/KeyRing/Project/Organization (least access preferred).
  """

    class EnvironmentValueValuesEnum(_messages.Enum):
        """The function environment the generated upload url will be used for.
    The upload url for 2nd Gen functions can also be used for 1st gen
    functions, but not vice versa. If not specified, 2nd generation-style
    upload URLs are generated.

    Values:
      ENVIRONMENT_UNSPECIFIED: Unspecified
      GEN_1: Gen 1
      GEN_2: Gen 2
    """
        ENVIRONMENT_UNSPECIFIED = 0
        GEN_1 = 1
        GEN_2 = 2
    environment = _messages.EnumField('EnvironmentValueValuesEnum', 1)
    kmsKeyName = _messages.StringField(2)