from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataSamplingReport(_messages.Message):
    """Contains per-worker telemetry about the data sampling feature.

  Fields:
    bytesWrittenDelta: Optional. Delta of bytes written to file from previous
      report.
    elementsSampledBytes: Optional. Delta of bytes sampled from previous
      report.
    elementsSampledCount: Optional. Delta of number of elements sampled from
      previous report.
    exceptionsSampledCount: Optional. Delta of number of samples taken from
      user code exceptions from previous report.
    pcollectionsSampledCount: Optional. Delta of number of PCollections
      sampled from previous report.
    persistenceErrorsCount: Optional. Delta of errors counts from persisting
      the samples from previous report.
    translationErrorsCount: Optional. Delta of errors counts from retrieving,
      or translating the samples from previous report.
  """
    bytesWrittenDelta = _messages.IntegerField(1)
    elementsSampledBytes = _messages.IntegerField(2)
    elementsSampledCount = _messages.IntegerField(3)
    exceptionsSampledCount = _messages.IntegerField(4)
    pcollectionsSampledCount = _messages.IntegerField(5)
    persistenceErrorsCount = _messages.IntegerField(6)
    translationErrorsCount = _messages.IntegerField(7)