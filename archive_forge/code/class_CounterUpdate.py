from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CounterUpdate(_messages.Message):
    """An update to a Counter sent from a worker.

  Fields:
    boolean: Boolean value for And, Or.
    cumulative: True if this counter is reported as the total cumulative
      aggregate value accumulated since the worker started working on this
      WorkItem. By default this is false, indicating that this counter is
      reported as a delta.
    distribution: Distribution data
    floatingPoint: Floating point value for Sum, Max, Min.
    floatingPointList: List of floating point numbers, for Set.
    floatingPointMean: Floating point mean aggregation value for Mean.
    integer: Integer value for Sum, Max, Min.
    integerGauge: Gauge data
    integerList: List of integers, for Set.
    integerMean: Integer mean aggregation value for Mean.
    internal: Value for internally-defined counters used by the Dataflow
      service.
    nameAndKind: Counter name and aggregation type.
    shortId: The service-generated short identifier for this counter. The
      short_id -> (name, metadata) mapping is constant for the lifetime of a
      job.
    stringList: List of strings, for Set.
    structuredNameAndMetadata: Counter structured name and metadata.
  """
    boolean = _messages.BooleanField(1)
    cumulative = _messages.BooleanField(2)
    distribution = _messages.MessageField('DistributionUpdate', 3)
    floatingPoint = _messages.FloatField(4)
    floatingPointList = _messages.MessageField('FloatingPointList', 5)
    floatingPointMean = _messages.MessageField('FloatingPointMean', 6)
    integer = _messages.MessageField('SplitInt64', 7)
    integerGauge = _messages.MessageField('IntegerGauge', 8)
    integerList = _messages.MessageField('IntegerList', 9)
    integerMean = _messages.MessageField('IntegerMean', 10)
    internal = _messages.MessageField('extra_types.JsonValue', 11)
    nameAndKind = _messages.MessageField('NameAndKind', 12)
    shortId = _messages.IntegerField(13)
    stringList = _messages.MessageField('StringList', 14)
    structuredNameAndMetadata = _messages.MessageField('CounterStructuredNameAndMetadata', 15)