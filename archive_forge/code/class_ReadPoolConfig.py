from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReadPoolConfig(_messages.Message):
    """Configuration for a read pool instance.

  Fields:
    nodeCount: Read capacity, i.e. number of nodes in a read pool instance.
  """
    nodeCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)