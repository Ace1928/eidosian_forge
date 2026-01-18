from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.faultinjectiontesting.v1alpha1 import faultinjectiontesting_v1alpha1_messages as messages
class ProjectsLocationsFaultsService(base_api.BaseApiService):
    """Service class for the projects_locations_faults resource."""
    _NAME = 'projects_locations_faults'

    def __init__(self, client):
        super(FaultinjectiontestingV1alpha1.ProjectsLocationsFaultsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Fault in a given project and location.

      Args:
        request: (FaultinjectiontestingProjectsLocationsFaultsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Fault) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/faults', http_method='POST', method_id='faultinjectiontesting.projects.locations.faults.create', ordered_params=['parent'], path_params=['parent'], query_params=['faultId', 'requestId'], relative_path='v1alpha1/{+parent}/faults', request_field='fault', request_type_name='FaultinjectiontestingProjectsLocationsFaultsCreateRequest', response_type_name='Fault', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Fault.

      Args:
        request: (FaultinjectiontestingProjectsLocationsFaultsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/faults/{faultsId}', http_method='DELETE', method_id='faultinjectiontesting.projects.locations.faults.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='FaultinjectiontestingProjectsLocationsFaultsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Fault.

      Args:
        request: (FaultinjectiontestingProjectsLocationsFaultsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Fault) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/faults/{faultsId}', http_method='GET', method_id='faultinjectiontesting.projects.locations.faults.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='FaultinjectiontestingProjectsLocationsFaultsGetRequest', response_type_name='Fault', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Faults in a given project and location.

      Args:
        request: (FaultinjectiontestingProjectsLocationsFaultsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFaultsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/faults', http_method='GET', method_id='faultinjectiontesting.projects.locations.faults.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/faults', request_field='', request_type_name='FaultinjectiontestingProjectsLocationsFaultsListRequest', response_type_name='ListFaultsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Fault.

      Args:
        request: (FaultinjectiontestingProjectsLocationsFaultsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Fault) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/faults/{faultsId}', http_method='PATCH', method_id='faultinjectiontesting.projects.locations.faults.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha1/{+name}', request_field='fault', request_type_name='FaultinjectiontestingProjectsLocationsFaultsPatchRequest', response_type_name='Fault', supports_download=False)