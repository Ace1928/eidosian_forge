from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2PrivacyMetric(_messages.Message):
    """Privacy metric to compute for reidentification risk analysis.

  Fields:
    categoricalStatsConfig: Categorical stats
    deltaPresenceEstimationConfig: delta-presence
    kAnonymityConfig: K-anonymity
    kMapEstimationConfig: k-map
    lDiversityConfig: l-diversity
    numericalStatsConfig: Numerical stats
  """
    categoricalStatsConfig = _messages.MessageField('GooglePrivacyDlpV2CategoricalStatsConfig', 1)
    deltaPresenceEstimationConfig = _messages.MessageField('GooglePrivacyDlpV2DeltaPresenceEstimationConfig', 2)
    kAnonymityConfig = _messages.MessageField('GooglePrivacyDlpV2KAnonymityConfig', 3)
    kMapEstimationConfig = _messages.MessageField('GooglePrivacyDlpV2KMapEstimationConfig', 4)
    lDiversityConfig = _messages.MessageField('GooglePrivacyDlpV2LDiversityConfig', 5)
    numericalStatsConfig = _messages.MessageField('GooglePrivacyDlpV2NumericalStatsConfig', 6)