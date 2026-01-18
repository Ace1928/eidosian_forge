from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConcatPosition(_messages.Message):
    """A position that encapsulates an inner position and an index for the
  inner position. A ConcatPosition can be used by a reader of a source that
  encapsulates a set of other sources.

  Fields:
    index: Index of the inner source.
    position: Position within the inner source.
  """
    index = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    position = _messages.MessageField('Position', 2)