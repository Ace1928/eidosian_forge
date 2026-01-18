from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.transferappliance.v1alpha1 import transferappliance_v1alpha1_messages as messages
class ProjectsLocationsOrdersService(base_api.BaseApiService):
    """Service class for the projects_locations_orders resource."""
    _NAME = 'projects_locations_orders'

    def __init__(self, client):
        super(TransferapplianceV1alpha1.ProjectsLocationsOrdersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Order in a given project and location.

      Args:
        request: (TransferapplianceProjectsLocationsOrdersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/orders', http_method='POST', method_id='transferappliance.projects.locations.orders.create', ordered_params=['parent'], path_params=['parent'], query_params=['orderId', 'requestId', 'validateOnly'], relative_path='v1alpha1/{+parent}/orders', request_field='order', request_type_name='TransferapplianceProjectsLocationsOrdersCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Order.

      Args:
        request: (TransferapplianceProjectsLocationsOrdersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/orders/{ordersId}', http_method='DELETE', method_id='transferappliance.projects.locations.orders.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'requestId', 'validateOnly'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='TransferapplianceProjectsLocationsOrdersDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Order.

      Args:
        request: (TransferapplianceProjectsLocationsOrdersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Order) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/orders/{ordersId}', http_method='GET', method_id='transferappliance.projects.locations.orders.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='TransferapplianceProjectsLocationsOrdersGetRequest', response_type_name='Order', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Orders in a given project and location.

      Args:
        request: (TransferapplianceProjectsLocationsOrdersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOrdersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/orders', http_method='GET', method_id='transferappliance.projects.locations.orders.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/orders', request_field='', request_type_name='TransferapplianceProjectsLocationsOrdersListRequest', response_type_name='ListOrdersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Order.

      Args:
        request: (TransferapplianceProjectsLocationsOrdersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/orders/{ordersId}', http_method='PATCH', method_id='transferappliance.projects.locations.orders.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'requestId', 'updateMask', 'validateOnly'], relative_path='v1alpha1/{+name}', request_field='order', request_type_name='TransferapplianceProjectsLocationsOrdersPatchRequest', response_type_name='Operation', supports_download=False)

    def Submit(self, request, global_params=None):
        """Submit an Order, moving it from the DRAFT state to PREPARING and updating any appliances associated with the order by moving them from the DRAFT state to ACTIVE. This method will attempt to set and validate any required permissions for a workload's service accounts on the workload's resources (e.g. KMS key, Cloud Storage bucket) for all appliances associated with the order. The caller must have the appropriate permissions to manage permissions for these resources.

      Args:
        request: (TransferapplianceProjectsLocationsOrdersSubmitRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Submit')
        return self._RunMethod(config, request, global_params=global_params)
    Submit.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/orders/{ordersId}:submit', http_method='POST', method_id='transferappliance.projects.locations.orders.submit', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:submit', request_field='submitOrderRequest', request_type_name='TransferapplianceProjectsLocationsOrdersSubmitRequest', response_type_name='Operation', supports_download=False)