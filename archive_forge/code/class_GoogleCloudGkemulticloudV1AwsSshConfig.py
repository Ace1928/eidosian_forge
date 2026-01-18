from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AwsSshConfig(_messages.Message):
    """SSH configuration for AWS resources.

  Fields:
    ec2KeyPair: Required. The name of the EC2 key pair used to login into
      cluster machines.
  """
    ec2KeyPair = _messages.StringField(1)