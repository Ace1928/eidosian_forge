from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpSaasplatformInsightsV1alphaConfiguredInsightResponse(_messages.Message):
    """The response for the configured insight.

  Fields:
    appliedConfig: Output only. Applied insight config to generate the result
      data rows.
    nextPageToken: Output only. Next page token to be fetched. Set to empty or
      NULL if there are no more pages available.
    rows: Output only. Result rows returned containing the required value(s)
      for configured insight.
  """
    appliedConfig = _messages.MessageField('GoogleCloudBeyondcorpSaasplatformInsightsV1alphaAppliedConfig', 1)
    nextPageToken = _messages.StringField(2)
    rows = _messages.MessageField('GoogleCloudBeyondcorpSaasplatformInsightsV1alphaRow', 3, repeated=True)