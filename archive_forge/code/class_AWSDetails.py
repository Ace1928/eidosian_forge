from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AWSDetails(_messages.Message):
    """Additional information for an asset fetched from AWS

  Fields:
    awsAccount: The AWS Account in [ARN format]
      (https://docs.aws.amazon.com/service-authorization/latest/reference/list
      _awsaccountmanagement.html#awsaccountmanagement-resources-for-iam-
      policies)
  """
    awsAccount = _messages.StringField(1)