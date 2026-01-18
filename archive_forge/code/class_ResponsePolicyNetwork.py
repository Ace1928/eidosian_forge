from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResponsePolicyNetwork(_messages.Message):
    """A ResponsePolicyNetwork object.

  Fields:
    kind: A string attribute.
    networkUrl: The fully qualified URL of the VPC network to bind to. This
      should be formatted like https://www.googleapis.com/compute/v1/projects/
      {project}/global/networks/{network}
  """
    kind = _messages.StringField(1, default='dns#responsePolicyNetwork')
    networkUrl = _messages.StringField(2)