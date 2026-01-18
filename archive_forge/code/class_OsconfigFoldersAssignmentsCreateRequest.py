from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigFoldersAssignmentsCreateRequest(_messages.Message):
    """A OsconfigFoldersAssignmentsCreateRequest object.

  Fields:
    assignment: A Assignment resource to be passed as the request body.
    parent: The resource name of the parent.
  """
    assignment = _messages.MessageField('Assignment', 1)
    parent = _messages.StringField(2, required=True)