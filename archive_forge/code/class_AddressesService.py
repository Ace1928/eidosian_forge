from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class AddressesService(base_api.BaseApiService):
    """Service class for the addresses resource."""
    _NAME = 'addresses'

    def __init__(self, client):
        super(ComputeBeta.AddressesService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of addresses. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeAddressesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AddressAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.addresses.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/addresses', request_field='', request_type_name='ComputeAddressesAggregatedListRequest', response_type_name='AddressAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified address resource.

      Args:
        request: (ComputeAddressesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.addresses.delete', ordered_params=['project', 'region', 'address'], path_params=['address', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/addresses/{address}', request_field='', request_type_name='ComputeAddressesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified address resource.

      Args:
        request: (ComputeAddressesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Address) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.addresses.get', ordered_params=['project', 'region', 'address'], path_params=['address', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/addresses/{address}', request_field='', request_type_name='ComputeAddressesGetRequest', response_type_name='Address', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates an address resource in the specified project by using the data included in the request.

      Args:
        request: (ComputeAddressesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.addresses.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/addresses', request_field='address', request_type_name='ComputeAddressesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of addresses contained within the specified region.

      Args:
        request: (ComputeAddressesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AddressList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.addresses.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/addresses', request_field='', request_type_name='ComputeAddressesListRequest', response_type_name='AddressList', supports_download=False)

    def Move(self, request, global_params=None):
        """Moves the specified address resource.

      Args:
        request: (ComputeAddressesMoveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Move')
        return self._RunMethod(config, request, global_params=global_params)
    Move.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.addresses.move', ordered_params=['project', 'region', 'address'], path_params=['address', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/addresses/{address}/move', request_field='regionAddressesMoveRequest', request_type_name='ComputeAddressesMoveRequest', response_type_name='Operation', supports_download=False)

    def SetLabels(self, request, global_params=None):
        """Sets the labels on an Address. To learn more about labels, read the Labeling Resources documentation.

      Args:
        request: (ComputeAddressesSetLabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetLabels')
        return self._RunMethod(config, request, global_params=global_params)
    SetLabels.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.addresses.setLabels', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/addresses/{resource}/setLabels', request_field='regionSetLabelsRequest', request_type_name='ComputeAddressesSetLabelsRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeAddressesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.addresses.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/addresses/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeAddressesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)