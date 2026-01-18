from __future__ import absolute_import
from apitools.base.py import base_api
from samples.servicemanagement_sample.servicemanagement_v1 import servicemanagement_v1_messages as messages
from the newest to the oldest.
class ServicesService(base_api.BaseApiService):
    """Service class for the services resource."""
    _NAME = u'services'

    def __init__(self, client):
        super(ServicemanagementV1.ServicesService, self).__init__(client)
        self._upload_configs = {}

    def ConvertConfig(self, request, global_params=None):
        """DEPRECATED. `SubmitConfigSource` with `validate_only=true` will provide.
config conversion moving forward.

Converts an API specification (e.g. Swagger spec) to an
equivalent `google.api.Service`.

      Args:
        request: (ConvertConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConvertConfigResponse) The response message.
      """
        config = self.GetMethodConfig('ConvertConfig')
        return self._RunMethod(config, request, global_params=global_params)
    ConvertConfig.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'servicemanagement.services.convertConfig', ordered_params=[], path_params=[], query_params=[], relative_path=u'v1/services:convertConfig', request_field='<request>', request_type_name=u'ConvertConfigRequest', response_type_name=u'ConvertConfigResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new managed service.

Operation<response: ManagedService>

      Args:
        request: (ManagedService) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'servicemanagement.services.create', ordered_params=[], path_params=[], query_params=[], relative_path=u'v1/services', request_field='<request>', request_type_name=u'ManagedService', response_type_name=u'Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a managed service.

Operation<response: google.protobuf.Empty>

      Args:
        request: (ServicemanagementServicesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'servicemanagement.services.delete', ordered_params=[u'serviceName'], path_params=[u'serviceName'], query_params=[], relative_path=u'v1/services/{serviceName}', request_field='', request_type_name=u'ServicemanagementServicesDeleteRequest', response_type_name=u'Operation', supports_download=False)

    def Disable(self, request, global_params=None):
        """Disable a managed service for a project.
Google Service Management will only disable the managed service even if
there are other services depend on the managed service.

Operation<response: DisableServiceResponse>

      Args:
        request: (ServicemanagementServicesDisableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Disable')
        return self._RunMethod(config, request, global_params=global_params)
    Disable.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'servicemanagement.services.disable', ordered_params=[u'serviceName'], path_params=[u'serviceName'], query_params=[], relative_path=u'v1/services/{serviceName}:disable', request_field=u'disableServiceRequest', request_type_name=u'ServicemanagementServicesDisableRequest', response_type_name=u'Operation', supports_download=False)

    def Enable(self, request, global_params=None):
        """Enable a managed service for a project with default setting.
If the managed service has dependencies, they will be enabled as well.

Operation<response: EnableServiceResponse>

      Args:
        request: (ServicemanagementServicesEnableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Enable')
        return self._RunMethod(config, request, global_params=global_params)
    Enable.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'servicemanagement.services.enable', ordered_params=[u'serviceName'], path_params=[u'serviceName'], query_params=[], relative_path=u'v1/services/{serviceName}:enable', request_field=u'enableServiceRequest', request_type_name=u'ServicemanagementServicesEnableRequest', response_type_name=u'Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a managed service. If the `consumer_project_id` is specified,.
the project's settings for the specified service are also returned.

      Args:
        request: (ServicemanagementServicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagedService) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'servicemanagement.services.get', ordered_params=[u'serviceName'], path_params=[u'serviceName'], query_params=[u'consumerProjectId', u'expand', u'view'], relative_path=u'v1/services/{serviceName}', request_field='', request_type_name=u'ServicemanagementServicesGetRequest', response_type_name=u'ManagedService', supports_download=False)

    def GetAccessPolicy(self, request, global_params=None):
        """Producer method to retrieve current policy.

      Args:
        request: (ServicemanagementServicesGetAccessPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAccessPolicy) The response message.
      """
        config = self.GetMethodConfig('GetAccessPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetAccessPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'servicemanagement.services.getAccessPolicy', ordered_params=[u'serviceName'], path_params=[u'serviceName'], query_params=[], relative_path=u'v1/services/{serviceName}/accessPolicy', request_field='', request_type_name=u'ServicemanagementServicesGetAccessPolicyRequest', response_type_name=u'ServiceAccessPolicy', supports_download=False)

    def GetConfig(self, request, global_params=None):
        """Gets a service config (version) for a managed service. If `config_id` is.
not specified, the latest service config will be returned.

      Args:
        request: (ServicemanagementServicesGetConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Service) The response message.
      """
        config = self.GetMethodConfig('GetConfig')
        return self._RunMethod(config, request, global_params=global_params)
    GetConfig.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'servicemanagement.services.getConfig', ordered_params=[u'serviceName'], path_params=[u'serviceName'], query_params=[u'configId'], relative_path=u'v1/services/{serviceName}/config', request_field='', request_type_name=u'ServicemanagementServicesGetConfigRequest', response_type_name=u'Service', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all managed services. If the `consumer_project_id` is specified,.
the project's settings for the specified service are also returned.

      Args:
        request: (ServicemanagementServicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServicesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'servicemanagement.services.list', ordered_params=[], path_params=[], query_params=[u'category', u'consumerProjectId', u'expand', u'pageSize', u'pageToken', u'producerProjectId'], relative_path=u'v1/services', request_field='', request_type_name=u'ServicemanagementServicesListRequest', response_type_name=u'ListServicesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified subset of the configuration. If the specified service.
does not exists the patch operation fails.

Operation<response: ManagedService>

      Args:
        request: (ServicemanagementServicesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'servicemanagement.services.patch', ordered_params=[u'serviceName'], path_params=[u'serviceName'], query_params=[u'updateMask'], relative_path=u'v1/services/{serviceName}', request_field=u'managedService', request_type_name=u'ServicemanagementServicesPatchRequest', response_type_name=u'Operation', supports_download=False)

    def PatchConfig(self, request, global_params=None):
        """Updates the specified subset of the service resource. Equivalent to.
calling `PatchService` with only the `service_config` field updated.

Operation<response: google.api.Service>

      Args:
        request: (ServicemanagementServicesPatchConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('PatchConfig')
        return self._RunMethod(config, request, global_params=global_params)
    PatchConfig.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'servicemanagement.services.patchConfig', ordered_params=[u'serviceName'], path_params=[u'serviceName'], query_params=[u'updateMask'], relative_path=u'v1/services/{serviceName}/config', request_field=u'service', request_type_name=u'ServicemanagementServicesPatchConfigRequest', response_type_name=u'Operation', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the configuration of a service.  If the specified service does not.
already exist, then it is created.

Operation<response: ManagedService>

      Args:
        request: (ServicemanagementServicesUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'servicemanagement.services.update', ordered_params=[u'serviceName'], path_params=[u'serviceName'], query_params=[u'updateMask'], relative_path=u'v1/services/{serviceName}', request_field=u'managedService', request_type_name=u'ServicemanagementServicesUpdateRequest', response_type_name=u'Operation', supports_download=False)

    def UpdateAccessPolicy(self, request, global_params=None):
        """Producer method to update the current policy.  This method will return an.
error if the policy is too large (more than 50 entries across all lists).

      Args:
        request: (ServiceAccessPolicy) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAccessPolicy) The response message.
      """
        config = self.GetMethodConfig('UpdateAccessPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateAccessPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'servicemanagement.services.updateAccessPolicy', ordered_params=[u'serviceName'], path_params=[u'serviceName'], query_params=[], relative_path=u'v1/services/{serviceName}/accessPolicy', request_field='<request>', request_type_name=u'ServiceAccessPolicy', response_type_name=u'ServiceAccessPolicy', supports_download=False)

    def UpdateConfig(self, request, global_params=None):
        """Updates the specified subset of the service resource. Equivalent to.
calling `UpdateService` with only the `service_config` field updated.

Operation<response: google.api.Service>

      Args:
        request: (ServicemanagementServicesUpdateConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('UpdateConfig')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateConfig.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'servicemanagement.services.updateConfig', ordered_params=[u'serviceName'], path_params=[u'serviceName'], query_params=[u'updateMask'], relative_path=u'v1/services/{serviceName}/config', request_field=u'service', request_type_name=u'ServicemanagementServicesUpdateConfigRequest', response_type_name=u'Operation', supports_download=False)