from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
class ProjectsRegionsService(base_api.BaseApiService):
    """Service class for the projects_regions resource."""
    _NAME = 'projects_regions'

    def __init__(self, client):
        super(DataprocV1.ProjectsRegionsService, self).__init__(client)
        self._upload_configs = {}