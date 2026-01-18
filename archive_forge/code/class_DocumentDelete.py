from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentDelete(_messages.Message):
    """A Document has been deleted. May be the result of multiple writes,
  including updates, the last of which deleted the Document. Multiple
  DocumentDelete messages may be returned for the same logical delete, if
  multiple targets are affected.

  Fields:
    document: The resource name of the Document that was deleted.
    readTime: The read timestamp at which the delete was observed. Greater or
      equal to the `commit_time` of the delete.
    removedTargetIds: A set of target IDs for targets that previously matched
      this entity.
  """
    document = _messages.StringField(1)
    readTime = _messages.StringField(2)
    removedTargetIds = _messages.IntegerField(3, repeated=True, variant=_messages.Variant.INT32)