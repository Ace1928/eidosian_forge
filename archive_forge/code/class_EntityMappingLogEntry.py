from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EntityMappingLogEntry(_messages.Message):
    """A single record of a rule which was used for a mapping.

  Fields:
    mappingComment: Comment.
    ruleId: Which rule caused this log entry.
    ruleRevisionId: Rule revision ID.
  """
    mappingComment = _messages.StringField(1)
    ruleId = _messages.StringField(2)
    ruleRevisionId = _messages.StringField(3)