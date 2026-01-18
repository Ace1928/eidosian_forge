from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApproximateSplitRequest(_messages.Message):
    """A suggestion by the service to the worker to dynamically split the
  WorkItem.

  Fields:
    fractionConsumed: A fraction at which to split the work item, from 0.0
      (beginning of the input) to 1.0 (end of the input).
    fractionOfRemainder: The fraction of the remainder of work to split the
      work item at, from 0.0 (split at the current position) to 1.0 (end of
      the input).
    position: A Position at which to split the work item.
  """
    fractionConsumed = _messages.FloatField(1)
    fractionOfRemainder = _messages.FloatField(2)
    position = _messages.MessageField('Position', 3)