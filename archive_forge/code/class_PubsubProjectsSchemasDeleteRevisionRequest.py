from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSchemasDeleteRevisionRequest(_messages.Message):
    """A PubsubProjectsSchemasDeleteRevisionRequest object.

  Fields:
    name: Required. The name of the schema revision to be deleted, with a
      revision ID explicitly included. Example: `projects/123/schemas/my-
      schema@c7cfa2a8`
    revisionId: Optional. This field is deprecated and should not be used for
      specifying the revision ID. The revision ID should be specified via the
      `name` parameter.
  """
    name = _messages.StringField(1, required=True)
    revisionId = _messages.StringField(2)