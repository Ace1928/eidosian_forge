from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CounterMetadata(_messages.Message):
    """CounterMetadata includes all static non-name non-value counter
  attributes.

  Enums:
    KindValueValuesEnum: Counter aggregation kind.
    StandardUnitsValueValuesEnum: System defined Units, see above enum.

  Fields:
    description: Human-readable description of the counter semantics.
    kind: Counter aggregation kind.
    otherUnits: A string referring to the unit type.
    standardUnits: System defined Units, see above enum.
  """

    class KindValueValuesEnum(_messages.Enum):
        """Counter aggregation kind.

    Values:
      INVALID: Counter aggregation kind was not set.
      SUM: Aggregated value is the sum of all contributed values.
      MAX: Aggregated value is the max of all contributed values.
      MIN: Aggregated value is the min of all contributed values.
      MEAN: Aggregated value is the mean of all contributed values.
      OR: Aggregated value represents the logical 'or' of all contributed
        values.
      AND: Aggregated value represents the logical 'and' of all contributed
        values.
      SET: Aggregated value is a set of unique contributed values.
      DISTRIBUTION: Aggregated value captures statistics about a distribution.
      LATEST_VALUE: Aggregated value tracks the latest value of a variable.
    """
        INVALID = 0
        SUM = 1
        MAX = 2
        MIN = 3
        MEAN = 4
        OR = 5
        AND = 6
        SET = 7
        DISTRIBUTION = 8
        LATEST_VALUE = 9

    class StandardUnitsValueValuesEnum(_messages.Enum):
        """System defined Units, see above enum.

    Values:
      BYTES: Counter returns a value in bytes.
      BYTES_PER_SEC: Counter returns a value in bytes per second.
      MILLISECONDS: Counter returns a value in milliseconds.
      MICROSECONDS: Counter returns a value in microseconds.
      NANOSECONDS: Counter returns a value in nanoseconds.
      TIMESTAMP_MSEC: Counter returns a timestamp in milliseconds.
      TIMESTAMP_USEC: Counter returns a timestamp in microseconds.
      TIMESTAMP_NSEC: Counter returns a timestamp in nanoseconds.
    """
        BYTES = 0
        BYTES_PER_SEC = 1
        MILLISECONDS = 2
        MICROSECONDS = 3
        NANOSECONDS = 4
        TIMESTAMP_MSEC = 5
        TIMESTAMP_USEC = 6
        TIMESTAMP_NSEC = 7
    description = _messages.StringField(1)
    kind = _messages.EnumField('KindValueValuesEnum', 2)
    otherUnits = _messages.StringField(3)
    standardUnits = _messages.EnumField('StandardUnitsValueValuesEnum', 4)