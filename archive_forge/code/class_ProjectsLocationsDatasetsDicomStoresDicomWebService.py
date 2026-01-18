from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1 import healthcare_v1_messages as messages
class ProjectsLocationsDatasetsDicomStoresDicomWebService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_dicomStores_dicomWeb resource."""
    _NAME = 'projects_locations_datasets_dicomStores_dicomWeb'

    def __init__(self, client):
        super(HealthcareV1.ProjectsLocationsDatasetsDicomStoresDicomWebService, self).__init__(client)
        self._upload_configs = {}