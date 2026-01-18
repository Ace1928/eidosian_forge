from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecommenderProjectsLocationsInsightTypesGetConfigRequest(_messages.Message):
    """A RecommenderProjectsLocationsInsightTypesGetConfigRequest object.

  Fields:
    name: Required. Name of the InsightTypeConfig to get. Acceptable formats:
      * `projects/[PROJECT_NUMBER]/locations/[LOCATION]/insightTypes/[INSIGHT_
      TYPE_ID]/config` * `projects/[PROJECT_ID]/locations/[LOCATION]/insightTy
      pes/[INSIGHT_TYPE_ID]/config` * `organizations/[ORGANIZATION_ID]/locatio
      ns/[LOCATION]/insightTypes/[INSIGHT_TYPE_ID]/config` * `billingAccounts/
      [BILLING_ACCOUNT_ID]/locations/[LOCATION]/insightTypes/[INSIGHT_TYPE_ID]
      /config`
  """
    name = _messages.StringField(1, required=True)