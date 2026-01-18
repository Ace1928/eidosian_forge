from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StageSource(_messages.Message):
    """Description of an input or output of an execution stage.

  Fields:
    name: Dataflow service generated name for this source.
    originalTransformOrCollection: User name for the original user transform
      or collection with which this source is most closely associated.
    sizeBytes: Size of the source, if measurable.
    userName: Human-readable name for this source; may be user or system
      generated.
  """
    name = _messages.StringField(1)
    originalTransformOrCollection = _messages.StringField(2)
    sizeBytes = _messages.IntegerField(3)
    userName = _messages.StringField(4)