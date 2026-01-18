from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2RequestedRiskAnalysisOptions(_messages.Message):
    """Risk analysis options.

  Fields:
    jobConfig: The job config for the risk job.
  """
    jobConfig = _messages.MessageField('GooglePrivacyDlpV2RiskAnalysisJobConfig', 1)