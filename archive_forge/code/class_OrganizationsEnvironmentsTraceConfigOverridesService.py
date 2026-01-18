from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsTraceConfigOverridesService(base_api.BaseApiService):
    """Service class for the organizations_environments_traceConfig_overrides resource."""
    _NAME = 'organizations_environments_traceConfig_overrides'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsTraceConfigOverridesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a trace configuration override. The response contains a system-generated UUID, that can be used to view, update, or delete the configuration override. Use the List API to view the existing trace configuration overrides.

      Args:
        request: (ApigeeOrganizationsEnvironmentsTraceConfigOverridesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1TraceConfigOverride) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/traceConfig/overrides', http_method='POST', method_id='apigee.organizations.environments.traceConfig.overrides.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/overrides', request_field='googleCloudApigeeV1TraceConfigOverride', request_type_name='ApigeeOrganizationsEnvironmentsTraceConfigOverridesCreateRequest', response_type_name='GoogleCloudApigeeV1TraceConfigOverride', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a distributed trace configuration override.

      Args:
        request: (ApigeeOrganizationsEnvironmentsTraceConfigOverridesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/traceConfig/overrides/{overridesId}', http_method='DELETE', method_id='apigee.organizations.environments.traceConfig.overrides.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsTraceConfigOverridesDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a trace configuration override.

      Args:
        request: (ApigeeOrganizationsEnvironmentsTraceConfigOverridesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1TraceConfigOverride) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/traceConfig/overrides/{overridesId}', http_method='GET', method_id='apigee.organizations.environments.traceConfig.overrides.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsTraceConfigOverridesGetRequest', response_type_name='GoogleCloudApigeeV1TraceConfigOverride', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all of the distributed trace configuration overrides in an environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsTraceConfigOverridesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListTraceConfigOverridesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/traceConfig/overrides', http_method='GET', method_id='apigee.organizations.environments.traceConfig.overrides.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/overrides', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsTraceConfigOverridesListRequest', response_type_name='GoogleCloudApigeeV1ListTraceConfigOverridesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a distributed trace configuration override. Note that the repeated fields have replace semantics when included in the field mask and that they will be overwritten by the value of the fields in the request body.

      Args:
        request: (ApigeeOrganizationsEnvironmentsTraceConfigOverridesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1TraceConfigOverride) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/traceConfig/overrides/{overridesId}', http_method='PATCH', method_id='apigee.organizations.environments.traceConfig.overrides.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudApigeeV1TraceConfigOverride', request_type_name='ApigeeOrganizationsEnvironmentsTraceConfigOverridesPatchRequest', response_type_name='GoogleCloudApigeeV1TraceConfigOverride', supports_download=False)