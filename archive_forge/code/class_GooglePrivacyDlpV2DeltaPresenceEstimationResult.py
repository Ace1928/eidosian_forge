from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DeltaPresenceEstimationResult(_messages.Message):
    """Result of the \\u03b4-presence computation. Note that these results are
  an estimation, not exact values.

  Fields:
    deltaPresenceEstimationHistogram: The intervals [min_probability,
      max_probability) do not overlap. If a value doesn't correspond to any
      such interval, the associated frequency is zero. For example, the
      following records: {min_probability: 0, max_probability: 0.1, frequency:
      17} {min_probability: 0.2, max_probability: 0.3, frequency: 42}
      {min_probability: 0.3, max_probability: 0.4, frequency: 99} mean that
      there are no record with an estimated probability in [0.1, 0.2) nor
      larger or equal to 0.4.
  """
    deltaPresenceEstimationHistogram = _messages.MessageField('GooglePrivacyDlpV2DeltaPresenceEstimationHistogramBucket', 1, repeated=True)