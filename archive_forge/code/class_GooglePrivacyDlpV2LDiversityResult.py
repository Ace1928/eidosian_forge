from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2LDiversityResult(_messages.Message):
    """Result of the l-diversity computation.

  Fields:
    sensitiveValueFrequencyHistogramBuckets: Histogram of l-diversity
      equivalence class sensitive value frequencies.
  """
    sensitiveValueFrequencyHistogramBuckets = _messages.MessageField('GooglePrivacyDlpV2LDiversityHistogramBucket', 1, repeated=True)