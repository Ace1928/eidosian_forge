from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1 import healthcare_v1_messages as messages
class ProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesFramesService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_dicomStores_studies_series_instances_frames resource."""
    _NAME = 'projects_locations_datasets_dicomStores_studies_series_instances_frames'

    def __init__(self, client):
        super(HealthcareV1.ProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesFramesService, self).__init__(client)
        self._upload_configs = {}

    def RetrieveFrames(self, request, global_params=None):
        """RetrieveFrames returns instances associated with the given study, series, SOP Instance UID and frame numbers. See [RetrieveTransaction] (http://dicom.nema.org/medical/dicom/current/output/html/part18.html#sect_10.4}. For details on the implementation of RetrieveFrames, see [DICOM frames](https://cloud.google.com/healthcare/docs/dicom#dicom_frames) in the Cloud Healthcare API conformance statement. For samples that show how to call RetrieveFrames, see [Retrieving DICOM data](https://cloud.google.com/healthcare/docs/how-tos/dicomweb#retrieving_dicom_data).

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesFramesRetrieveFramesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('RetrieveFrames')
        return self._RunMethod(config, request, global_params=global_params)
    RetrieveFrames.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}/dicomWeb/studies/{studiesId}/series/{seriesId}/instances/{instancesId}/frames/{framesId}', http_method='GET', method_id='healthcare.projects.locations.datasets.dicomStores.studies.series.instances.frames.retrieveFrames', ordered_params=['parent', 'dicomWebPath'], path_params=['dicomWebPath', 'parent'], query_params=[], relative_path='v1/{+parent}/dicomWeb/{+dicomWebPath}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesFramesRetrieveFramesRequest', response_type_name='HttpBody', supports_download=False)

    def RetrieveRendered(self, request, global_params=None):
        """RetrieveRenderedFrames returns instances associated with the given study, series, SOP Instance UID and frame numbers in an acceptable Rendered Media Type. See [RetrieveTransaction] (http://dicom.nema.org/medical/dicom/current/output/html/part18.html#sect_10.4). For details on the implementation of RetrieveRenderedFrames, see [Rendered resources](https://cloud.google.com/healthcare/docs/dicom#rendered_resources) in the Cloud Healthcare API conformance statement. For samples that show how to call RetrieveRenderedFrames, see [Retrieving consumer image formats](https://cloud.google.com/healthcare/docs/how-tos/dicomweb#retrieving_consumer_image_formats).

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesFramesRetrieveRenderedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('RetrieveRendered')
        return self._RunMethod(config, request, global_params=global_params)
    RetrieveRendered.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}/dicomWeb/studies/{studiesId}/series/{seriesId}/instances/{instancesId}/frames/{framesId}/rendered', http_method='GET', method_id='healthcare.projects.locations.datasets.dicomStores.studies.series.instances.frames.retrieveRendered', ordered_params=['parent', 'dicomWebPath'], path_params=['dicomWebPath', 'parent'], query_params=[], relative_path='v1/{+parent}/dicomWeb/{+dicomWebPath}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesFramesRetrieveRenderedRequest', response_type_name='HttpBody', supports_download=False)