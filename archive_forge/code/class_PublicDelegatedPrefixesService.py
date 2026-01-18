from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class PublicDelegatedPrefixesService(base_api.BaseApiService):
    """Service class for the publicDelegatedPrefixes resource."""
    _NAME = 'publicDelegatedPrefixes'

    def __init__(self, client):
        super(ComputeBeta.PublicDelegatedPrefixesService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Lists all PublicDelegatedPrefix resources owned by the specific project across all scopes. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputePublicDelegatedPrefixesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PublicDelegatedPrefixAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.publicDelegatedPrefixes.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/publicDelegatedPrefixes', request_field='', request_type_name='ComputePublicDelegatedPrefixesAggregatedListRequest', response_type_name='PublicDelegatedPrefixAggregatedList', supports_download=False)

    def Announce(self, request, global_params=None):
        """Announces the specified PublicDelegatedPrefix in the given region.

      Args:
        request: (ComputePublicDelegatedPrefixesAnnounceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Announce')
        return self._RunMethod(config, request, global_params=global_params)
    Announce.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.publicDelegatedPrefixes.announce', ordered_params=['project', 'region', 'publicDelegatedPrefix'], path_params=['project', 'publicDelegatedPrefix', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/publicDelegatedPrefixes/{publicDelegatedPrefix}/announce', request_field='', request_type_name='ComputePublicDelegatedPrefixesAnnounceRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified PublicDelegatedPrefix in the given region.

      Args:
        request: (ComputePublicDelegatedPrefixesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.publicDelegatedPrefixes.delete', ordered_params=['project', 'region', 'publicDelegatedPrefix'], path_params=['project', 'publicDelegatedPrefix', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/publicDelegatedPrefixes/{publicDelegatedPrefix}', request_field='', request_type_name='ComputePublicDelegatedPrefixesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified PublicDelegatedPrefix resource in the given region.

      Args:
        request: (ComputePublicDelegatedPrefixesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PublicDelegatedPrefix) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.publicDelegatedPrefixes.get', ordered_params=['project', 'region', 'publicDelegatedPrefix'], path_params=['project', 'publicDelegatedPrefix', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/publicDelegatedPrefixes/{publicDelegatedPrefix}', request_field='', request_type_name='ComputePublicDelegatedPrefixesGetRequest', response_type_name='PublicDelegatedPrefix', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a PublicDelegatedPrefix in the specified project in the given region using the parameters that are included in the request.

      Args:
        request: (ComputePublicDelegatedPrefixesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.publicDelegatedPrefixes.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/publicDelegatedPrefixes', request_field='publicDelegatedPrefix', request_type_name='ComputePublicDelegatedPrefixesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the PublicDelegatedPrefixes for a project in the given region.

      Args:
        request: (ComputePublicDelegatedPrefixesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PublicDelegatedPrefixList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.publicDelegatedPrefixes.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/publicDelegatedPrefixes', request_field='', request_type_name='ComputePublicDelegatedPrefixesListRequest', response_type_name='PublicDelegatedPrefixList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified PublicDelegatedPrefix resource with the data included in the request. This method supports PATCH semantics and uses JSON merge patch format and processing rules.

      Args:
        request: (ComputePublicDelegatedPrefixesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.publicDelegatedPrefixes.patch', ordered_params=['project', 'region', 'publicDelegatedPrefix'], path_params=['project', 'publicDelegatedPrefix', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/publicDelegatedPrefixes/{publicDelegatedPrefix}', request_field='publicDelegatedPrefixResource', request_type_name='ComputePublicDelegatedPrefixesPatchRequest', response_type_name='Operation', supports_download=False)

    def Withdraw(self, request, global_params=None):
        """Withdraws the specified PublicDelegatedPrefix in the given region.

      Args:
        request: (ComputePublicDelegatedPrefixesWithdrawRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Withdraw')
        return self._RunMethod(config, request, global_params=global_params)
    Withdraw.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.publicDelegatedPrefixes.withdraw', ordered_params=['project', 'region', 'publicDelegatedPrefix'], path_params=['project', 'publicDelegatedPrefix', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/publicDelegatedPrefixes/{publicDelegatedPrefix}/withdraw', request_field='', request_type_name='ComputePublicDelegatedPrefixesWithdrawRequest', response_type_name='Operation', supports_download=False)