from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicenetworking.v1 import servicenetworking_v1_messages as messages
class ServicesRolesService(base_api.BaseApiService):
    """Service class for the services_roles resource."""
    _NAME = 'services_roles'

    def __init__(self, client):
        super(ServicenetworkingV1.ServicesRolesService, self).__init__(client)
        self._upload_configs = {}

    def Add(self, request, global_params=None):
        """Service producers can use this method to add roles in the shared VPC host project. Each role is bound to the provided member. Each role must be selected from within an allowlisted set of roles. Each role is applied at only the granularity specified in the allowlist.

      Args:
        request: (ServicenetworkingServicesRolesAddRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Add')
        return self._RunMethod(config, request, global_params=global_params)
    Add.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/services/{servicesId}/roles:add', http_method='POST', method_id='servicenetworking.services.roles.add', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/roles:add', request_field='addRolesRequest', request_type_name='ServicenetworkingServicesRolesAddRequest', response_type_name='Operation', supports_download=False)