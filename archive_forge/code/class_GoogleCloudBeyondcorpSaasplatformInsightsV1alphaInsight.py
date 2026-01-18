from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpSaasplatformInsightsV1alphaInsight(_messages.Message):
    """The Insight object with configuration that was returned and actual list
  of records.

  Fields:
    appliedConfig: Output only. Applied insight config to generate the result
      data rows.
    metadata: Output only. Metadata for the Insight.
    name: Output only. The insight resource name. e.g. `organizations/{organiz
      ation_id}/locations/{location_id}/insights/{insight_id}` OR
      `projects/{project_id}/locations/{location_id}/insights/{insight_id}`.
    rows: Output only. Result rows returned containing the required value(s).
  """
    appliedConfig = _messages.MessageField('GoogleCloudBeyondcorpSaasplatformInsightsV1alphaAppliedConfig', 1)
    metadata = _messages.MessageField('GoogleCloudBeyondcorpSaasplatformInsightsV1alphaInsightMetadata', 2)
    name = _messages.StringField(3)
    rows = _messages.MessageField('GoogleCloudBeyondcorpSaasplatformInsightsV1alphaRow', 4, repeated=True)