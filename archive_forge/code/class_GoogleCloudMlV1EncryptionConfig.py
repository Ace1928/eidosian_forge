from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1EncryptionConfig(_messages.Message):
    """Represents a custom encryption key configuration that can be applied to
  a resource.

  Fields:
    kmsKeyName: The Cloud KMS resource identifier of the customer-managed
      encryption key used to protect a resource, such as a training job. It
      has the following format: `projects/{PROJECT_ID}/locations/{REGION}/keyR
      ings/{KEY_RING_NAME}/cryptoKeys/{KEY_NAME}`
  """
    kmsKeyName = _messages.StringField(1)