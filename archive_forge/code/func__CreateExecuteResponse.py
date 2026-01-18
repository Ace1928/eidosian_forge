from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_projector
import six
def _CreateExecuteResponse(client, messages, request, is_async, command_prefix):
    """Creates an ExecutePatchJobResponse message."""
    async_response = client.projects_patchJobs.Execute(request)
    patch_job_name = osconfig_command_utils.GetResourceName(async_response.name)
    if is_async:
        log.status.Print('Execution in progress for patch job [{}]'.format(patch_job_name))
        log.status.Print('Run the [{} describe] command to check the status of this execution.'.format(command_prefix))
        return async_response
    patch_job_poller = osconfig_api_utils.Poller(client, messages)
    get_request = messages.OsconfigProjectsPatchJobsGetRequest(name=async_response.name)
    sync_response = waiter.WaitFor(patch_job_poller, get_request, custom_tracker=_CreateProgressTracker(patch_job_name), tracker_update_func=_UpdateProgressTracker, pre_start_sleep_ms=5000, exponential_sleep_multiplier=1, sleep_ms=5000)
    log.status.Print('Execution for patch job [{}] has completed with status [{}].'.format(patch_job_name, sync_response.state))
    log.status.Print('Run the [{} list-instance-details] command to view any instance failure reasons.'.format(command_prefix))
    return sync_response