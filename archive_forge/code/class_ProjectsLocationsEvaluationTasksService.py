from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
class ProjectsLocationsEvaluationTasksService(base_api.BaseApiService):
    """Service class for the projects_locations_evaluationTasks resource."""
    _NAME = 'projects_locations_evaluationTasks'

    def __init__(self, client):
        super(AiplatformV1beta1.ProjectsLocationsEvaluationTasksService, self).__init__(client)
        self._upload_configs = {}