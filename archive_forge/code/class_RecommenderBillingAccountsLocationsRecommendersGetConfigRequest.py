from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecommenderBillingAccountsLocationsRecommendersGetConfigRequest(_messages.Message):
    """A RecommenderBillingAccountsLocationsRecommendersGetConfigRequest
  object.

  Fields:
    name: Required. Name of the Recommendation Config to get. Acceptable
      formats: * `projects/[PROJECT_NUMBER]/locations/[LOCATION]/recommenders/
      [RECOMMENDER_ID]/config` * `projects/[PROJECT_ID]/locations/[LOCATION]/r
      ecommenders/[RECOMMENDER_ID]/config` * `organizations/[ORGANIZATION_ID]/
      locations/[LOCATION]/recommenders/[RECOMMENDER_ID]/config` * `billingAcc
      ounts/[BILLING_ACCOUNT_ID]/locations/[LOCATION]/recommenders/[RECOMMENDE
      R_ID]/config`
  """
    name = _messages.StringField(1, required=True)