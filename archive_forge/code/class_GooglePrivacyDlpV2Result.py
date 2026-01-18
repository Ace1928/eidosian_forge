from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Result(_messages.Message):
    """All result fields mentioned below are updated while the job is
  processing.

  Fields:
    hybridStats: Statistics related to the processing of hybrid inspect.
    infoTypeStats: Statistics of how many instances of each info type were
      found during inspect job.
    processedBytes: Total size in bytes that were processed.
    totalEstimatedBytes: Estimate of the number of bytes to process.
  """
    hybridStats = _messages.MessageField('GooglePrivacyDlpV2HybridInspectStatistics', 1)
    infoTypeStats = _messages.MessageField('GooglePrivacyDlpV2InfoTypeStats', 2, repeated=True)
    processedBytes = _messages.IntegerField(3)
    totalEstimatedBytes = _messages.IntegerField(4)