from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NumericIndexingSpecValue(_messages.Message):
    """Indexing spec for a numeric field.

    By default, only exact match
    queries will be supported for numeric fields. Setting the
    numericIndexingSpec allows range queries to be supported.

    Fields:
      maxValue: Maximum value of this field. This is meant to be indicative
        rather than enforced. Values outside this range will still be indexed,
        but search may not be as performant.
      minValue: Minimum value of this field. This is meant to be indicative
        rather than enforced. Values outside this range will still be indexed,
        but search may not be as performant.
    """
    maxValue = _messages.FloatField(1)
    minValue = _messages.FloatField(2)