from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomTrafficMatrix(_messages.Message):
    """Represents a custom traffic matrix passed directly by the calling
  client.

  Fields:
    shapeGeneratedEntry: List of distinct shape generators that describe the
      traffic matrix.
  """
    shapeGeneratedEntry = _messages.MessageField('ShapeGeneratedEntry', 1, repeated=True)