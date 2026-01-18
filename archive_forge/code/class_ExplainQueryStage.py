from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExplainQueryStage(_messages.Message):
    """A ExplainQueryStage object.

  Fields:
    computeRatioAvg: Relative amount of time the average shard spent on CPU-
      bound tasks.
    computeRatioMax: Relative amount of time the slowest shard spent on CPU-
      bound tasks.
    id: Unique ID for stage within plan.
    name: Human-readable name for stage.
    readRatioAvg: Relative amount of time the average shard spent reading
      input.
    readRatioMax: Relative amount of time the slowest shard spent reading
      input.
    recordsRead: Number of records read into the stage.
    recordsWritten: Number of records written by the stage.
    steps: List of operations within the stage in dependency order
      (approximately chronological).
    waitRatioAvg: Relative amount of time the average shard spent waiting to
      be scheduled.
    waitRatioMax: Relative amount of time the slowest shard spent waiting to
      be scheduled.
    writeRatioAvg: Relative amount of time the average shard spent on writing
      output.
    writeRatioMax: Relative amount of time the slowest shard spent on writing
      output.
  """
    computeRatioAvg = _messages.FloatField(1)
    computeRatioMax = _messages.FloatField(2)
    id = _messages.IntegerField(3)
    name = _messages.StringField(4)
    readRatioAvg = _messages.FloatField(5)
    readRatioMax = _messages.FloatField(6)
    recordsRead = _messages.IntegerField(7)
    recordsWritten = _messages.IntegerField(8)
    steps = _messages.MessageField('ExplainQueryStep', 9, repeated=True)
    waitRatioAvg = _messages.FloatField(10)
    waitRatioMax = _messages.FloatField(11)
    writeRatioAvg = _messages.FloatField(12)
    writeRatioMax = _messages.FloatField(13)