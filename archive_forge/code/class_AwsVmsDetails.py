from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AwsVmsDetails(_messages.Message):
    """AWSVmsDetails describes VMs in AWS.

  Fields:
    details: The details of the AWS VMs.
  """
    details = _messages.MessageField('AwsVmDetails', 1, repeated=True)