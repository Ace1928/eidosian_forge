from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1beta1 import healthcare_v1beta1_messages as messages
class ProjectsLocationsDatasetsDicomStoresDicomWebStudiesSeriesInstancesService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_dicomStores_dicomWeb_studies_series_instances resource."""
    _NAME = 'projects_locations_datasets_dicomStores_dicomWeb_studies_series_instances'

    def __init__(self, client):
        super(HealthcareV1beta1.ProjectsLocationsDatasetsDicomStoresDicomWebStudiesSeriesInstancesService, self).__init__(client)
        self._upload_configs = {}

    def GetStorageInfo(self, request, global_params=None):
        """GetStorageInfo returns the storage info of the specified resource.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresDicomWebStudiesSeriesInstancesGetStorageInfoRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StorageInfo) The response message.
      """
        config = self.GetMethodConfig('GetStorageInfo')
        return self._RunMethod(config, request, global_params=global_params)
    GetStorageInfo.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}/dicomWeb/studies/{studiesId}/series/{seriesId}/instances/{instancesId}:getStorageInfo', http_method='GET', method_id='healthcare.projects.locations.datasets.dicomStores.dicomWeb.studies.series.instances.getStorageInfo', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta1/{+resource}:getStorageInfo', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresDicomWebStudiesSeriesInstancesGetStorageInfoRequest', response_type_name='StorageInfo', supports_download=False)