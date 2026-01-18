from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def BillingAccountsInsightTypeInsightsService(api_version):
    """Returns the service class for the Billing Account insights."""
    client = RecommenderClient(api_version)
    return client.billingAccounts_locations_insightTypes_insights