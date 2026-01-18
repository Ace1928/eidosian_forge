from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AwsConfigEncryption(_messages.Message):
    """Config encryption for user data.

  Fields:
    kmsKeyArn: Required. The ARN of the AWS KMS key used to encrypt user data.
  """
    kmsKeyArn = _messages.StringField(1)