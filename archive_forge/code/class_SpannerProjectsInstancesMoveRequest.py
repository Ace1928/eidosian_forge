from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesMoveRequest(_messages.Message):
    """A SpannerProjectsInstancesMoveRequest object.

  Fields:
    moveInstanceRequest: A MoveInstanceRequest resource to be passed as the
      request body.
    name: Required. The instance to move. Values are of the form
      `projects//instances/`.
  """
    moveInstanceRequest = _messages.MessageField('MoveInstanceRequest', 1)
    name = _messages.StringField(2, required=True)