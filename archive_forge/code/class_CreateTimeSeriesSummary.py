from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateTimeSeriesSummary(_messages.Message):
    """Summary of the result of a failed request to write data to a time
  series.

  Fields:
    errors: The number of points that failed to be written. Order is not
      guaranteed.
    successPointCount: The number of points that were successfully written.
    totalPointCount: The number of points in the request.
  """
    errors = _messages.MessageField('Error', 1, repeated=True)
    successPointCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    totalPointCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)