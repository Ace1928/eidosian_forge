from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigOrganizationsAssignmentsPatchRequest(_messages.Message):
    """A OsconfigOrganizationsAssignmentsPatchRequest object.

  Fields:
    assignment: A Assignment resource to be passed as the request body.
    name: The resource name of the Assignment.
    updateMask: Field mask that controls which fields of the Assignment should
      be updated.
  """
    assignment = _messages.MessageField('Assignment', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)