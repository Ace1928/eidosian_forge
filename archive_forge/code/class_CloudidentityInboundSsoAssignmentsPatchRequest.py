from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityInboundSsoAssignmentsPatchRequest(_messages.Message):
    """A CloudidentityInboundSsoAssignmentsPatchRequest object.

  Fields:
    inboundSsoAssignment: A InboundSsoAssignment resource to be passed as the
      request body.
    name: Output only. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      Inbound SSO Assignment.
    updateMask: Required. The list of fields to be updated.
  """
    inboundSsoAssignment = _messages.MessageField('InboundSsoAssignment', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)