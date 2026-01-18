from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1beta1 import healthcare_v1beta1_messages as messages
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