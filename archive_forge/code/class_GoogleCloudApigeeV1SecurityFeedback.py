from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityFeedback(_messages.Message):
    """Represents a feedback report from an Advanced API Security customer.

  Fields:
    comment: Optional. Optional text the user can provide for additional,
      unstructured context.
    createTime: Output only. The time when this specific feedback id was
      created.
    feedbackContext: Required. One or more attribute/value pairs for
      constraining the feedback.
    name: Output only. Identifier. The feedback name is intended to be a
      system-generated uuid.
  """
    comment = _messages.StringField(1)
    createTime = _messages.StringField(2)
    feedbackContext = _messages.MessageField('GoogleCloudApigeeV1SecurityFeedbackFeedbackContext', 3, repeated=True)
    name = _messages.StringField(4)