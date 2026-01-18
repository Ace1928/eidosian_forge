from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecommenderOrganizationsLocationsInsightTypesUpdateConfigRequest(_messages.Message):
    """A RecommenderOrganizationsLocationsInsightTypesUpdateConfigRequest
  object.

  Fields:
    googleCloudRecommenderV1InsightTypeConfig: A
      GoogleCloudRecommenderV1InsightTypeConfig resource to be passed as the
      request body.
    name: Name of insight type config. Eg, projects/[PROJECT_NUMBER]/locations
      /[LOCATION]/insightTypes/[INSIGHT_TYPE_ID]/config
    updateMask: The list of fields to be updated.
    validateOnly: If true, validate the request and preview the change, but do
      not actually update it.
  """
    googleCloudRecommenderV1InsightTypeConfig = _messages.MessageField('GoogleCloudRecommenderV1InsightTypeConfig', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)