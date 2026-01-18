from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.binaryauthorization.v1 import binaryauthorization_v1_messages as messages
class ProjectsPlatformsGkeService(base_api.BaseApiService):
    """Service class for the projects_platforms_gke resource."""
    _NAME = 'projects_platforms_gke'

    def __init__(self, client):
        super(BinaryauthorizationV1.ProjectsPlatformsGkeService, self).__init__(client)
        self._upload_configs = {}