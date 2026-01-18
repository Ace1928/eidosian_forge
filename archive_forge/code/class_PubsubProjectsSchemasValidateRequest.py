from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSchemasValidateRequest(_messages.Message):
    """A PubsubProjectsSchemasValidateRequest object.

  Fields:
    parent: Required. The name of the project in which to validate schemas.
      Format is `projects/{project-id}`.
    validateSchemaRequest: A ValidateSchemaRequest resource to be passed as
      the request body.
  """
    parent = _messages.StringField(1, required=True)
    validateSchemaRequest = _messages.MessageField('ValidateSchemaRequest', 2)