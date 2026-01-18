from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1 import healthcare_v1_messages as messages
class ProjectsLocationsDatasetsDicomStoresStudiesService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_dicomStores_studies resource."""
    _NAME = 'projects_locations_datasets_dicomStores_studies'

    def __init__(self, client):
        super(HealthcareV1.ProjectsLocationsDatasetsDicomStoresStudiesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """DeleteStudy deletes all instances within the given study. Delete requests are equivalent to the GET requests specified in the Retrieve transaction. The method returns an Operation which will be marked successful when the deletion is complete. Warning: Instances cannot be inserted into a study that is being deleted by an operation until the operation completes. For samples that show how to call DeleteStudy, see [Deleting a study, series, or instance](https://cloud.google.com/healthcare/docs/how-tos/dicomweb#deleting_a_study_series_or_instance).

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresStudiesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}/dicomWeb/studies/{studiesId}', http_method='DELETE', method_id='healthcare.projects.locations.datasets.dicomStores.studies.delete', ordered_params=['parent', 'dicomWebPath'], path_params=['dicomWebPath', 'parent'], query_params=[], relative_path='v1/{+parent}/dicomWeb/{+dicomWebPath}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresStudiesDeleteRequest', response_type_name='Operation', supports_download=False)

    def RetrieveMetadata(self, request, global_params=None):
        """RetrieveStudyMetadata returns instance associated with the given study presented as metadata with the bulk data removed. See [RetrieveTransaction] (http://dicom.nema.org/medical/dicom/current/output/html/part18.html#sect_10.4). For details on the implementation of RetrieveStudyMetadata, see [Metadata resources](https://cloud.google.com/healthcare/docs/dicom#metadata_resources) in the Cloud Healthcare API conformance statement. For samples that show how to call RetrieveStudyMetadata, see [Retrieving metadata](https://cloud.google.com/healthcare/docs/how-tos/dicomweb#retrieving_metadata).

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresStudiesRetrieveMetadataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('RetrieveMetadata')
        return self._RunMethod(config, request, global_params=global_params)
    RetrieveMetadata.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}/dicomWeb/studies/{studiesId}/metadata', http_method='GET', method_id='healthcare.projects.locations.datasets.dicomStores.studies.retrieveMetadata', ordered_params=['parent', 'dicomWebPath'], path_params=['dicomWebPath', 'parent'], query_params=[], relative_path='v1/{+parent}/dicomWeb/{+dicomWebPath}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresStudiesRetrieveMetadataRequest', response_type_name='HttpBody', supports_download=False)

    def RetrieveStudy(self, request, global_params=None):
        """RetrieveStudy returns all instances within the given study. See [RetrieveTransaction] (http://dicom.nema.org/medical/dicom/current/output/html/part18.html#sect_10.4). For details on the implementation of RetrieveStudy, see [DICOM study/series/instances](https://cloud.google.com/healthcare/docs/dicom#dicom_studyseriesinstances) in the Cloud Healthcare API conformance statement. For samples that show how to call RetrieveStudy, see [Retrieving DICOM data](https://cloud.google.com/healthcare/docs/how-tos/dicomweb#retrieving_dicom_data).

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresStudiesRetrieveStudyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('RetrieveStudy')
        return self._RunMethod(config, request, global_params=global_params)
    RetrieveStudy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}/dicomWeb/studies/{studiesId}', http_method='GET', method_id='healthcare.projects.locations.datasets.dicomStores.studies.retrieveStudy', ordered_params=['parent', 'dicomWebPath'], path_params=['dicomWebPath', 'parent'], query_params=[], relative_path='v1/{+parent}/dicomWeb/{+dicomWebPath}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresStudiesRetrieveStudyRequest', response_type_name='HttpBody', supports_download=False)

    def SearchForInstances(self, request, global_params=None):
        """SearchForInstances returns a list of matching instances. See [Search Transaction] (http://dicom.nema.org/medical/dicom/current/output/html/part18.html#sect_10.6). For details on the implementation of SearchForInstances, see [Search transaction](https://cloud.google.com/healthcare/docs/dicom#search_transaction) in the Cloud Healthcare API conformance statement. For samples that show how to call SearchForInstances, see [Searching for studies, series, instances, and frames](https://cloud.google.com/healthcare/docs/how-tos/dicomweb#searching_for_studies_series_instances_and_frames).

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresStudiesSearchForInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('SearchForInstances')
        return self._RunMethod(config, request, global_params=global_params)
    SearchForInstances.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}/dicomWeb/studies/{studiesId}/instances', http_method='GET', method_id='healthcare.projects.locations.datasets.dicomStores.studies.searchForInstances', ordered_params=['parent', 'dicomWebPath'], path_params=['dicomWebPath', 'parent'], query_params=[], relative_path='v1/{+parent}/dicomWeb/{+dicomWebPath}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresStudiesSearchForInstancesRequest', response_type_name='HttpBody', supports_download=False)

    def SearchForSeries(self, request, global_params=None):
        """SearchForSeries returns a list of matching series. See [Search Transaction] (http://dicom.nema.org/medical/dicom/current/output/html/part18.html#sect_10.6). For details on the implementation of SearchForSeries, see [Search transaction](https://cloud.google.com/healthcare/docs/dicom#search_transaction) in the Cloud Healthcare API conformance statement. For samples that show how to call SearchForSeries, see [Searching for studies, series, instances, and frames](https://cloud.google.com/healthcare/docs/how-tos/dicomweb#searching_for_studies_series_instances_and_frames).

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresStudiesSearchForSeriesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('SearchForSeries')
        return self._RunMethod(config, request, global_params=global_params)
    SearchForSeries.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}/dicomWeb/studies/{studiesId}/series', http_method='GET', method_id='healthcare.projects.locations.datasets.dicomStores.studies.searchForSeries', ordered_params=['parent', 'dicomWebPath'], path_params=['dicomWebPath', 'parent'], query_params=[], relative_path='v1/{+parent}/dicomWeb/{+dicomWebPath}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresStudiesSearchForSeriesRequest', response_type_name='HttpBody', supports_download=False)

    def StoreInstances(self, request, global_params=None):
        """StoreInstances stores DICOM instances associated with study instance unique identifiers (SUID). See [Store Transaction] (http://dicom.nema.org/medical/dicom/current/output/html/part18.html#sect_10.5). For details on the implementation of StoreInstances, see [Store transaction](https://cloud.google.com/healthcare/docs/dicom#store_transaction) in the Cloud Healthcare API conformance statement. For samples that show how to call StoreInstances, see [Storing DICOM data](https://cloud.google.com/healthcare/docs/how-tos/dicomweb#storing_dicom_data).

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresStudiesStoreInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('StoreInstances')
        return self._RunMethod(config, request, global_params=global_params)
    StoreInstances.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}/dicomWeb/studies/{studiesId}', http_method='POST', method_id='healthcare.projects.locations.datasets.dicomStores.studies.storeInstances', ordered_params=['parent', 'dicomWebPath'], path_params=['dicomWebPath', 'parent'], query_params=[], relative_path='v1/{+parent}/dicomWeb/{+dicomWebPath}', request_field='httpBody', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresStudiesStoreInstancesRequest', response_type_name='HttpBody', supports_download=False)