from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
class ProjectsLocationsNfsSharesService(base_api.BaseApiService):
    """Service class for the projects_locations_nfsShares resource."""
    _NAME = 'projects_locations_nfsShares'

    def __init__(self, client):
        super(BaremetalsolutionV2.ProjectsLocationsNfsSharesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create an NFS share.

      Args:
        request: (BaremetalsolutionProjectsLocationsNfsSharesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/nfsShares', http_method='POST', method_id='baremetalsolution.projects.locations.nfsShares.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/nfsShares', request_field='nfsShare', request_type_name='BaremetalsolutionProjectsLocationsNfsSharesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete an NFS share. The underlying volume is automatically deleted.

      Args:
        request: (BaremetalsolutionProjectsLocationsNfsSharesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/nfsShares/{nfsSharesId}', http_method='DELETE', method_id='baremetalsolution.projects.locations.nfsShares.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BaremetalsolutionProjectsLocationsNfsSharesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Get details of a single NFS share.

      Args:
        request: (BaremetalsolutionProjectsLocationsNfsSharesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NfsShare) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/nfsShares/{nfsSharesId}', http_method='GET', method_id='baremetalsolution.projects.locations.nfsShares.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BaremetalsolutionProjectsLocationsNfsSharesGetRequest', response_type_name='NfsShare', supports_download=False)

    def List(self, request, global_params=None):
        """List NFS shares.

      Args:
        request: (BaremetalsolutionProjectsLocationsNfsSharesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNfsSharesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/nfsShares', http_method='GET', method_id='baremetalsolution.projects.locations.nfsShares.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/nfsShares', request_field='', request_type_name='BaremetalsolutionProjectsLocationsNfsSharesListRequest', response_type_name='ListNfsSharesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update details of a single NFS share.

      Args:
        request: (BaremetalsolutionProjectsLocationsNfsSharesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/nfsShares/{nfsSharesId}', http_method='PATCH', method_id='baremetalsolution.projects.locations.nfsShares.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='nfsShare', request_type_name='BaremetalsolutionProjectsLocationsNfsSharesPatchRequest', response_type_name='Operation', supports_download=False)

    def Rename(self, request, global_params=None):
        """RenameNfsShare sets a new name for an nfsshare. Use with caution, previous names become immediately invalidated.

      Args:
        request: (BaremetalsolutionProjectsLocationsNfsSharesRenameRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NfsShare) The response message.
      """
        config = self.GetMethodConfig('Rename')
        return self._RunMethod(config, request, global_params=global_params)
    Rename.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/nfsShares/{nfsSharesId}:rename', http_method='POST', method_id='baremetalsolution.projects.locations.nfsShares.rename', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:rename', request_field='renameNfsShareRequest', request_type_name='BaremetalsolutionProjectsLocationsNfsSharesRenameRequest', response_type_name='NfsShare', supports_download=False)