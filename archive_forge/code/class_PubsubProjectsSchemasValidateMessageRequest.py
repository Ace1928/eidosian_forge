from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSchemasValidateMessageRequest(_messages.Message):
    """A PubsubProjectsSchemasValidateMessageRequest object.

  Fields:
    parent: Required. The name of the project in which to validate schemas.
      Format is `projects/{project-id}`.
    validateMessageRequest: A ValidateMessageRequest resource to be passed as
      the request body.
  """
    parent = _messages.StringField(1, required=True)
    validateMessageRequest = _messages.MessageField('ValidateMessageRequest', 2)