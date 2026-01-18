from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1beta1 import healthcare_v1beta1_messages as messages
class ProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesBulkdataService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_dicomStores_studies_series_instances_bulkdata resource."""
    _NAME = 'projects_locations_datasets_dicomStores_studies_series_instances_bulkdata'

    def __init__(self, client):
        super(HealthcareV1beta1.ProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesBulkdataService, self).__init__(client)
        self._upload_configs = {}

    def RetrieveBulkdata(self, request, global_params=None):
        """Returns uncompressed, unencoded bytes representing the referenced bulkdata tag from an instance. See [Retrieve Transaction] (http://dicom.nema.org/medical/dicom/current/output/html/part18.html#sect_10.4){: .external}. For details on the implementation of RetrieveBulkdata, see [Bulkdata resources](https://cloud.google.com/healthcare/docs/dicom#bulkdata-resources) in the Cloud Healthcare API conformance statement. For samples that show how to call RetrieveBulkdata, see [Retrieve bulkdata](https://cloud.google.com/healthcare/docs/how-tos/dicomweb#retrieve-bulkdata).

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesBulkdataRetrieveBulkdataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('RetrieveBulkdata')
        return self._RunMethod(config, request, global_params=global_params)
    RetrieveBulkdata.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}/dicomWeb/studies/{studiesId}/series/{seriesId}/instances/{instancesId}/bulkdata/{bulkdataId}/{bulkdataId1}', http_method='GET', method_id='healthcare.projects.locations.datasets.dicomStores.studies.series.instances.bulkdata.retrieveBulkdata', ordered_params=['parent', 'dicomWebPath'], path_params=['dicomWebPath', 'parent'], query_params=[], relative_path='v1beta1/{+parent}/dicomWeb/{+dicomWebPath}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresStudiesSeriesInstancesBulkdataRetrieveBulkdataRequest', response_type_name='HttpBody', supports_download=False)