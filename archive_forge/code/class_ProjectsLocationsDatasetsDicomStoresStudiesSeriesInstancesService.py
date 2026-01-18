from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1 import healthcare_v1_messages as messages
class ProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_dicomStores_studies_series_instances resource."""
    _NAME = 'projects_locations_datasets_dicomStores_studies_series_instances'

    def __init__(self, client):
        super(HealthcareV1.ProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """DeleteInstance deletes an instance associated with the given study, series, and SOP Instance UID. Delete requests are equivalent to the GET requests specified in the Retrieve transaction. Study and series search results can take a few seconds to be updated after an instance is deleted using DeleteInstance. For samples that show how to call DeleteInstance, see [Deleting a study, series, or instance](https://cloud.google.com/healthcare/docs/how-tos/dicomweb#deleting_a_study_series_or_instance).

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}/dicomWeb/studies/{studiesId}/series/{seriesId}/instances/{instancesId}', http_method='DELETE', method_id='healthcare.projects.locations.datasets.dicomStores.studies.series.instances.delete', ordered_params=['parent', 'dicomWebPath'], path_params=['dicomWebPath', 'parent'], query_params=[], relative_path='v1/{+parent}/dicomWeb/{+dicomWebPath}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesDeleteRequest', response_type_name='Empty', supports_download=False)

    def RetrieveInstance(self, request, global_params=None):
        """RetrieveInstance returns instance associated with the given study, series, and SOP Instance UID. See [RetrieveTransaction] (http://dicom.nema.org/medical/dicom/current/output/html/part18.html#sect_10.4). For details on the implementation of RetrieveInstance, see [DICOM study/series/instances](https://cloud.google.com/healthcare/docs/dicom#dicom_studyseriesinstances) and [DICOM instances](https://cloud.google.com/healthcare/docs/dicom#dicom_instances) in the Cloud Healthcare API conformance statement. For samples that show how to call RetrieveInstance, see [Retrieving an instance](https://cloud.google.com/healthcare/docs/how-tos/dicomweb#retrieving_an_instance).

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesRetrieveInstanceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('RetrieveInstance')
        return self._RunMethod(config, request, global_params=global_params)
    RetrieveInstance.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}/dicomWeb/studies/{studiesId}/series/{seriesId}/instances/{instancesId}', http_method='GET', method_id='healthcare.projects.locations.datasets.dicomStores.studies.series.instances.retrieveInstance', ordered_params=['parent', 'dicomWebPath'], path_params=['dicomWebPath', 'parent'], query_params=[], relative_path='v1/{+parent}/dicomWeb/{+dicomWebPath}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesRetrieveInstanceRequest', response_type_name='HttpBody', supports_download=False)

    def RetrieveMetadata(self, request, global_params=None):
        """RetrieveInstanceMetadata returns instance associated with the given study, series, and SOP Instance UID presented as metadata with the bulk data removed. See [RetrieveTransaction] (http://dicom.nema.org/medical/dicom/current/output/html/part18.html#sect_10.4). For details on the implementation of RetrieveInstanceMetadata, see [Metadata resources](https://cloud.google.com/healthcare/docs/dicom#metadata_resources) in the Cloud Healthcare API conformance statement. For samples that show how to call RetrieveInstanceMetadata, see [Retrieving metadata](https://cloud.google.com/healthcare/docs/how-tos/dicomweb#retrieving_metadata).

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesRetrieveMetadataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('RetrieveMetadata')
        return self._RunMethod(config, request, global_params=global_params)
    RetrieveMetadata.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}/dicomWeb/studies/{studiesId}/series/{seriesId}/instances/{instancesId}/metadata', http_method='GET', method_id='healthcare.projects.locations.datasets.dicomStores.studies.series.instances.retrieveMetadata', ordered_params=['parent', 'dicomWebPath'], path_params=['dicomWebPath', 'parent'], query_params=[], relative_path='v1/{+parent}/dicomWeb/{+dicomWebPath}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesRetrieveMetadataRequest', response_type_name='HttpBody', supports_download=False)

    def RetrieveRendered(self, request, global_params=None):
        """RetrieveRenderedInstance returns instance associated with the given study, series, and SOP Instance UID in an acceptable Rendered Media Type. See [RetrieveTransaction] (http://dicom.nema.org/medical/dicom/current/output/html/part18.html#sect_10.4). For details on the implementation of RetrieveRenderedInstance, see [Rendered resources](https://cloud.google.com/healthcare/docs/dicom#rendered_resources) in the Cloud Healthcare API conformance statement. For samples that show how to call RetrieveRenderedInstance, see [Retrieving consumer image formats](https://cloud.google.com/healthcare/docs/how-tos/dicomweb#retrieving_consumer_image_formats).

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesRetrieveRenderedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('RetrieveRendered')
        return self._RunMethod(config, request, global_params=global_params)
    RetrieveRendered.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}/dicomWeb/studies/{studiesId}/series/{seriesId}/instances/{instancesId}/rendered', http_method='GET', method_id='healthcare.projects.locations.datasets.dicomStores.studies.series.instances.retrieveRendered', ordered_params=['parent', 'dicomWebPath'], path_params=['dicomWebPath', 'parent'], query_params=[], relative_path='v1/{+parent}/dicomWeb/{+dicomWebPath}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesRetrieveRenderedRequest', response_type_name='HttpBody', supports_download=False)