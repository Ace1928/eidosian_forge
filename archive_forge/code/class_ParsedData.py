from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ParsedData(_messages.Message):
    """The content of a HL7v2 message in a structured format.

  Fields:
    segments: A Segment attribute.
  """
    segments = _messages.MessageField('Segment', 1, repeated=True)