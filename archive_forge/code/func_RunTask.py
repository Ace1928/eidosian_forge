from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.command_lib.iam import iam_util
def RunTask(task_ref, args):
    """Runs dataplex task with input updates to execution spec args and labels."""
    run_task_req = dataplex_api.GetMessageModule().DataplexProjectsLocationsLakesTasksRunRequest(name=task_ref.RelativeName(), googleCloudDataplexV1RunTaskRequest=dataplex_api.GetMessageModule().GoogleCloudDataplexV1RunTaskRequest(labels=dataplex_api.CreateLabels(dataplex_api.GetMessageModule().GoogleCloudDataplexV1RunTaskRequest, args), args=CreateArgs(dataplex_api.GetMessageModule().GoogleCloudDataplexV1RunTaskRequest, args)))
    run_task_response = dataplex_api.GetClientInstance().projects_locations_lakes_tasks.Run(run_task_req)
    return run_task_response