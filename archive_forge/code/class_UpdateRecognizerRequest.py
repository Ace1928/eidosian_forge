from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateRecognizerRequest(_messages.Message):
    """Request message for the UpdateRecognizer method.

  Fields:
    recognizer: Required. The Recognizer to update. The Recognizer's `name`
      field is used to identify the Recognizer to update. Format:
      `projects/{project}/locations/{location}/recognizers/{recognizer}`.
    updateMask: The list of fields to update. If empty, all non-default valued
      fields are considered for update. Use `*` to update the entire
      Recognizer resource.
    validateOnly: If set, validate the request and preview the updated
      Recognizer, but do not actually update it.
  """
    recognizer = _messages.MessageField('Recognizer', 1)
    updateMask = _messages.StringField(2)
    validateOnly = _messages.BooleanField(3)