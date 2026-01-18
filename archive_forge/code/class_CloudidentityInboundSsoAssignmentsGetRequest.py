from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityInboundSsoAssignmentsGetRequest(_messages.Message):
    """A CloudidentityInboundSsoAssignmentsGetRequest object.

  Fields:
    name: Required. The [resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      InboundSsoAssignment to fetch. Format:
      `inboundSsoAssignments/{assignment}`
  """
    name = _messages.StringField(1, required=True)