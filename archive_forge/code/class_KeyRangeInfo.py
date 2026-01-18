from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeyRangeInfo(_messages.Message):
    """A message representing information for a key range (possibly one key).

  Fields:
    contextValues: The list of context values for this key range.
    endKeyIndex: The index of the end key in indexed_keys.
    info: Information about this key range, for all metrics.
    keysCount: The number of keys this range covers.
    metric: The name of the metric. e.g. "latency".
    startKeyIndex: The index of the start key in indexed_keys.
    timeOffset: The time offset. This is the time since the start of the time
      interval.
    unit: The unit of the metric. This is an unstructured field and will be
      mapped as is to the user.
    value: The value of the metric.
  """
    contextValues = _messages.MessageField('ContextValue', 1, repeated=True)
    endKeyIndex = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    info = _messages.MessageField('LocalizedString', 3)
    keysCount = _messages.IntegerField(4)
    metric = _messages.MessageField('LocalizedString', 5)
    startKeyIndex = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    timeOffset = _messages.StringField(7)
    unit = _messages.MessageField('LocalizedString', 8)
    value = _messages.FloatField(9, variant=_messages.Variant.FLOAT)