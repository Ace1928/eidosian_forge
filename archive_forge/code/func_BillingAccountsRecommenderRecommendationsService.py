from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def BillingAccountsRecommenderRecommendationsService(api_version):
    """Returns the service class for the Billing Account recommendations."""
    client = RecommenderClient(api_version)
    return client.billingAccounts_locations_recommenders_recommendations