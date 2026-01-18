from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dns.v1alpha2 import dns_v1alpha2_messages as messages
class ResponsePoliciesService(base_api.BaseApiService):
    """Service class for the responsePolicies resource."""
    _NAME = 'responsePolicies'

    def __init__(self, client):
        super(DnsV1alpha2.ResponsePoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Response Policy.

      Args:
        request: (DnsResponsePoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResponsePolicy) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='dns.responsePolicies.create', ordered_params=['project'], path_params=['project'], query_params=['clientOperationId'], relative_path='dns/v1alpha2/projects/{project}/responsePolicies', request_field='responsePolicy', request_type_name='DnsResponsePoliciesCreateRequest', response_type_name='ResponsePolicy', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a previously created Response Policy. Fails if the response policy is non-empty or still being referenced by a network.

      Args:
        request: (DnsResponsePoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DnsResponsePoliciesDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='dns.responsePolicies.delete', ordered_params=['project', 'responsePolicy'], path_params=['project', 'responsePolicy'], query_params=['clientOperationId'], relative_path='dns/v1alpha2/projects/{project}/responsePolicies/{responsePolicy}', request_field='', request_type_name='DnsResponsePoliciesDeleteRequest', response_type_name='DnsResponsePoliciesDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Fetches the representation of an existing Response Policy.

      Args:
        request: (DnsResponsePoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResponsePolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dns.responsePolicies.get', ordered_params=['project', 'responsePolicy'], path_params=['project', 'responsePolicy'], query_params=['clientOperationId'], relative_path='dns/v1alpha2/projects/{project}/responsePolicies/{responsePolicy}', request_field='', request_type_name='DnsResponsePoliciesGetRequest', response_type_name='ResponsePolicy', supports_download=False)

    def List(self, request, global_params=None):
        """Enumerates all Response Policies associated with a project.

      Args:
        request: (DnsResponsePoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResponsePoliciesListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dns.responsePolicies.list', ordered_params=['project'], path_params=['project'], query_params=['maxResults', 'pageToken'], relative_path='dns/v1alpha2/projects/{project}/responsePolicies', request_field='', request_type_name='DnsResponsePoliciesListRequest', response_type_name='ResponsePoliciesListResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Applies a partial update to an existing Response Policy.

      Args:
        request: (DnsResponsePoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResponsePoliciesPatchResponse) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='dns.responsePolicies.patch', ordered_params=['project', 'responsePolicy'], path_params=['project', 'responsePolicy'], query_params=['clientOperationId'], relative_path='dns/v1alpha2/projects/{project}/responsePolicies/{responsePolicy}', request_field='responsePolicyResource', request_type_name='DnsResponsePoliciesPatchRequest', response_type_name='ResponsePoliciesPatchResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates an existing Response Policy.

      Args:
        request: (DnsResponsePoliciesUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResponsePoliciesUpdateResponse) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='dns.responsePolicies.update', ordered_params=['project', 'responsePolicy'], path_params=['project', 'responsePolicy'], query_params=['clientOperationId'], relative_path='dns/v1alpha2/projects/{project}/responsePolicies/{responsePolicy}', request_field='responsePolicyResource', request_type_name='DnsResponsePoliciesUpdateRequest', response_type_name='ResponsePoliciesUpdateResponse', supports_download=False)