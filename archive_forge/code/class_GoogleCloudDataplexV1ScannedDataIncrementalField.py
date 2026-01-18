from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ScannedDataIncrementalField(_messages.Message):
    """A data range denoted by a pair of start/end values of a field.

  Fields:
    end: Value that marks the end of the range.
    field: The field that contains values which monotonically increases over
      time (e.g. a timestamp column).
    start: Value that marks the start of the range.
  """
    end = _messages.StringField(1)
    field = _messages.StringField(2)
    start = _messages.StringField(3)