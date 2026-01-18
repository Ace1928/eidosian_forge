from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1EncryptionSpec(_messages.Message):
    """Represents a customer-managed encryption key spec that can be applied to
  a top-level resource.

  Fields:
    kmsKeyName: Required. The Cloud KMS resource identifier of the customer
      managed encryption key used to protect a resource. Has the form:
      `projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-
      key`. The key needs to be in the same region as where the compute
      resource is created.
  """
    kmsKeyName = _messages.StringField(1)