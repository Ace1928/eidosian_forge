from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2AnalyzeDataSourceRiskDetails(_messages.Message):
    """Result of a risk analysis operation request.

  Fields:
    categoricalStatsResult: Categorical stats result
    deltaPresenceEstimationResult: Delta-presence result
    kAnonymityResult: K-anonymity result
    kMapEstimationResult: K-map result
    lDiversityResult: L-divesity result
    numericalStatsResult: Numerical stats result
    requestedOptions: The configuration used for this job.
    requestedPrivacyMetric: Privacy metric to compute.
    requestedSourceTable: Input dataset to compute metrics over.
  """
    categoricalStatsResult = _messages.MessageField('GooglePrivacyDlpV2CategoricalStatsResult', 1)
    deltaPresenceEstimationResult = _messages.MessageField('GooglePrivacyDlpV2DeltaPresenceEstimationResult', 2)
    kAnonymityResult = _messages.MessageField('GooglePrivacyDlpV2KAnonymityResult', 3)
    kMapEstimationResult = _messages.MessageField('GooglePrivacyDlpV2KMapEstimationResult', 4)
    lDiversityResult = _messages.MessageField('GooglePrivacyDlpV2LDiversityResult', 5)
    numericalStatsResult = _messages.MessageField('GooglePrivacyDlpV2NumericalStatsResult', 6)
    requestedOptions = _messages.MessageField('GooglePrivacyDlpV2RequestedRiskAnalysisOptions', 7)
    requestedPrivacyMetric = _messages.MessageField('GooglePrivacyDlpV2PrivacyMetric', 8)
    requestedSourceTable = _messages.MessageField('GooglePrivacyDlpV2BigQueryTable', 9)