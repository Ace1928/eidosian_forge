from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1 import healthcare_v1_messages as messages
class ProjectsLocationsDatasetsDicomStoresDicomWebStudiesSeriesService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_dicomStores_dicomWeb_studies_series resource."""
    _NAME = 'projects_locations_datasets_dicomStores_dicomWeb_studies_series'

    def __init__(self, client):
        super(HealthcareV1.ProjectsLocationsDatasetsDicomStoresDicomWebStudiesSeriesService, self).__init__(client)
        self._upload_configs = {}

    def GetSeriesMetrics(self, request, global_params=None):
        """GetSeriesMetrics returns metrics for a series.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresDicomWebStudiesSeriesGetSeriesMetricsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SeriesMetrics) The response message.
      """
        config = self.GetMethodConfig('GetSeriesMetrics')
        return self._RunMethod(config, request, global_params=global_params)
    GetSeriesMetrics.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}/dicomWeb/studies/{studiesId}/series/{seriesId}:getSeriesMetrics', http_method='GET', method_id='healthcare.projects.locations.datasets.dicomStores.dicomWeb.studies.series.getSeriesMetrics', ordered_params=['series'], path_params=['series'], query_params=[], relative_path='v1/{+series}:getSeriesMetrics', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresDicomWebStudiesSeriesGetSeriesMetricsRequest', response_type_name='SeriesMetrics', supports_download=False)