from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CommitSchemaRequest(_messages.Message):
    """Request for CommitSchema method.

  Fields:
    schema: Required. The schema revision to commit.
  """
    schema = _messages.MessageField('Schema', 1)