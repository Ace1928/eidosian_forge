from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateRecognizerRequest(_messages.Message):
    """Request message for the CreateRecognizer method.

  Fields:
    parent: Required. The project and location where this Recognizer will be
      created. The expected format is
      `projects/{project}/locations/{location}`.
    recognizer: Required. The Recognizer to create.
    recognizerId: The ID to use for the Recognizer, which will become the
      final component of the Recognizer's resource name. This value should be
      4-63 characters, and valid characters are /a-z-/.
    validateOnly: If set, validate the request and preview the Recognizer, but
      do not actually create it.
  """
    parent = _messages.StringField(1)
    recognizer = _messages.MessageField('Recognizer', 2)
    recognizerId = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)