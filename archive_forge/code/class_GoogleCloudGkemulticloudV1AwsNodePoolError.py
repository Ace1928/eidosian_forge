from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AwsNodePoolError(_messages.Message):
    """AwsNodePoolError describes errors found on AWS node pools.

  Fields:
    message: Human-friendly description of the error.
  """
    message = _messages.StringField(1)