from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AwsSecurityGroup(_messages.Message):
    """AwsSecurityGroup describes a security group of an AWS VM.

  Fields:
    id: The AWS security group id.
    name: The AWS security group name.
  """
    id = _messages.StringField(1)
    name = _messages.StringField(2)