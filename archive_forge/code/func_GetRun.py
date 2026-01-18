from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def GetRun(project, region, run_id, run_type):
    """Get a PipelineRun/TaskRun."""
    client = GetClientInstance()
    messages = GetMessagesModule()
    if run_type == 'pipelinerun':
        pipeline_run_resource = resources.REGISTRY.Parse(run_id, collection='cloudbuild.projects.locations.pipelineRuns', api_version='v2', params={'projectsId': project, 'locationsId': region, 'pipelineRunsId': run_id})
        pipeline_run = client.projects_locations_pipelineRuns.Get(messages.CloudbuildProjectsLocationsPipelineRunsGetRequest(name=pipeline_run_resource.RelativeName()))
        return pipeline_run
    elif run_type == 'taskrun':
        task_run_resource = resources.REGISTRY.Parse(run_id, collection='cloudbuild.projects.locations.taskRuns', api_version='v2', params={'projectsId': project, 'locationsId': region, 'taskRunsId': run_id})
        task_run = client.projects_locations_taskRuns.Get(messages.CloudbuildProjectsLocationsTaskRunsGetRequest(name=task_run_resource.RelativeName()))
        return task_run