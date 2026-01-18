from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class PublicAdvertisedPrefixesService(base_api.BaseApiService):
    """Service class for the publicAdvertisedPrefixes resource."""
    _NAME = 'publicAdvertisedPrefixes'

    def __init__(self, client):
        super(ComputeBeta.PublicAdvertisedPrefixesService, self).__init__(client)
        self._upload_configs = {}

    def Announce(self, request, global_params=None):
        """Announces the specified PublicAdvertisedPrefix.

      Args:
        request: (ComputePublicAdvertisedPrefixesAnnounceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Announce')
        return self._RunMethod(config, request, global_params=global_params)
    Announce.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.publicAdvertisedPrefixes.announce', ordered_params=['project', 'publicAdvertisedPrefix'], path_params=['project', 'publicAdvertisedPrefix'], query_params=['requestId'], relative_path='projects/{project}/global/publicAdvertisedPrefixes/{publicAdvertisedPrefix}/announce', request_field='', request_type_name='ComputePublicAdvertisedPrefixesAnnounceRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified PublicAdvertisedPrefix.

      Args:
        request: (ComputePublicAdvertisedPrefixesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.publicAdvertisedPrefixes.delete', ordered_params=['project', 'publicAdvertisedPrefix'], path_params=['project', 'publicAdvertisedPrefix'], query_params=['requestId'], relative_path='projects/{project}/global/publicAdvertisedPrefixes/{publicAdvertisedPrefix}', request_field='', request_type_name='ComputePublicAdvertisedPrefixesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified PublicAdvertisedPrefix resource.

      Args:
        request: (ComputePublicAdvertisedPrefixesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PublicAdvertisedPrefix) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.publicAdvertisedPrefixes.get', ordered_params=['project', 'publicAdvertisedPrefix'], path_params=['project', 'publicAdvertisedPrefix'], query_params=[], relative_path='projects/{project}/global/publicAdvertisedPrefixes/{publicAdvertisedPrefix}', request_field='', request_type_name='ComputePublicAdvertisedPrefixesGetRequest', response_type_name='PublicAdvertisedPrefix', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a PublicAdvertisedPrefix in the specified project using the parameters that are included in the request.

      Args:
        request: (ComputePublicAdvertisedPrefixesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.publicAdvertisedPrefixes.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/publicAdvertisedPrefixes', request_field='publicAdvertisedPrefix', request_type_name='ComputePublicAdvertisedPrefixesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the PublicAdvertisedPrefixes for a project.

      Args:
        request: (ComputePublicAdvertisedPrefixesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PublicAdvertisedPrefixList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.publicAdvertisedPrefixes.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/publicAdvertisedPrefixes', request_field='', request_type_name='ComputePublicAdvertisedPrefixesListRequest', response_type_name='PublicAdvertisedPrefixList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified Router resource with the data included in the request. This method supports PATCH semantics and uses JSON merge patch format and processing rules.

      Args:
        request: (ComputePublicAdvertisedPrefixesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.publicAdvertisedPrefixes.patch', ordered_params=['project', 'publicAdvertisedPrefix'], path_params=['project', 'publicAdvertisedPrefix'], query_params=['requestId'], relative_path='projects/{project}/global/publicAdvertisedPrefixes/{publicAdvertisedPrefix}', request_field='publicAdvertisedPrefixResource', request_type_name='ComputePublicAdvertisedPrefixesPatchRequest', response_type_name='Operation', supports_download=False)

    def Withdraw(self, request, global_params=None):
        """Withdraws the specified PublicAdvertisedPrefix.

      Args:
        request: (ComputePublicAdvertisedPrefixesWithdrawRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Withdraw')
        return self._RunMethod(config, request, global_params=global_params)
    Withdraw.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.publicAdvertisedPrefixes.withdraw', ordered_params=['project', 'publicAdvertisedPrefix'], path_params=['project', 'publicAdvertisedPrefix'], query_params=['requestId'], relative_path='projects/{project}/global/publicAdvertisedPrefixes/{publicAdvertisedPrefix}/withdraw', request_field='', request_type_name='ComputePublicAdvertisedPrefixesWithdrawRequest', response_type_name='Operation', supports_download=False)