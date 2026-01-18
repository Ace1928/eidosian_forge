from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recommender.v1alpha2 import recommender_v1alpha2_messages as messages
class BillingAccountsService(base_api.BaseApiService):
    """Service class for the billingAccounts resource."""
    _NAME = 'billingAccounts'

    def __init__(self, client):
        super(RecommenderV1alpha2.BillingAccountsService, self).__init__(client)
        self._upload_configs = {}