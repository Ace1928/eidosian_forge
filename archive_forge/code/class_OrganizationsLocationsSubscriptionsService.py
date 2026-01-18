from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.beyondcorp.v1alpha import beyondcorp_v1alpha_messages as messages
class OrganizationsLocationsSubscriptionsService(base_api.BaseApiService):
    """Service class for the organizations_locations_subscriptions resource."""
    _NAME = 'organizations_locations_subscriptions'

    def __init__(self, client):
        super(BeyondcorpV1alpha.OrganizationsLocationsSubscriptionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new BeyondCorp Enterprise Subscription in a given organization. Location will always be global as BeyondCorp subscriptions are per organization.

      Args:
        request: (BeyondcorpOrganizationsLocationsSubscriptionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpSaasplatformSubscriptionsV1alphaSubscription) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/{locationsId}/subscriptions', http_method='POST', method_id='beyondcorp.organizations.locations.subscriptions.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha/{+parent}/subscriptions', request_field='googleCloudBeyondcorpSaasplatformSubscriptionsV1alphaSubscription', request_type_name='BeyondcorpOrganizationsLocationsSubscriptionsCreateRequest', response_type_name='GoogleCloudBeyondcorpSaasplatformSubscriptionsV1alphaSubscription', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Subscription.

      Args:
        request: (BeyondcorpOrganizationsLocationsSubscriptionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpSaasplatformSubscriptionsV1alphaSubscription) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/{locationsId}/subscriptions/{subscriptionsId}', http_method='GET', method_id='beyondcorp.organizations.locations.subscriptions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='BeyondcorpOrganizationsLocationsSubscriptionsGetRequest', response_type_name='GoogleCloudBeyondcorpSaasplatformSubscriptionsV1alphaSubscription', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Subscriptions in a given organization and location.

      Args:
        request: (BeyondcorpOrganizationsLocationsSubscriptionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpSaasplatformSubscriptionsV1alphaListSubscriptionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/{locationsId}/subscriptions', http_method='GET', method_id='beyondcorp.organizations.locations.subscriptions.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/subscriptions', request_field='', request_type_name='BeyondcorpOrganizationsLocationsSubscriptionsListRequest', response_type_name='GoogleCloudBeyondcorpSaasplatformSubscriptionsV1alphaListSubscriptionsResponse', supports_download=False)