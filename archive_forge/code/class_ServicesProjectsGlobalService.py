from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicenetworking.v1 import servicenetworking_v1_messages as messages
class ServicesProjectsGlobalService(base_api.BaseApiService):
    """Service class for the services_projects_global resource."""
    _NAME = 'services_projects_global'

    def __init__(self, client):
        super(ServicenetworkingV1.ServicesProjectsGlobalService, self).__init__(client)
        self._upload_configs = {}