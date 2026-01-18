from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import dataclasses
import functools
import random
import string
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.run import condition as run_condition
from googlecloudsdk.api_lib.run import configuration
from googlecloudsdk.api_lib.run import domain_mapping
from googlecloudsdk.api_lib.run import execution
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run import job
from googlecloudsdk.api_lib.run import metric_names
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import route
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import task
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.run import artifact_registry
from googlecloudsdk.command_lib.run import config_changes as config_changes_mod
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import messages_util
from googlecloudsdk.command_lib.run import name_generator
from googlecloudsdk.command_lib.run import op_pollers
from googlecloudsdk.command_lib.run import resource_name_conversion
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.run.sourcedeploys import deployer
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
import six
def RunJob(self, job_ref, tracker=None, wait=False, asyn=False, release_track=None, overrides=None):
    """Run a Cloud Run Job, creating an Execution.

    Args:
      job_ref: Resource, the job to run
      tracker: StagedProgressTracker, to report on the progress of running
      wait: boolean, True to wait until the job is complete
      asyn: bool, if True, return without waiting for anything
      release_track: ReleaseTrack, the release track of a command calling this
      overrides: ExecutionOverrides to be applied for this run of a job

    Returns:
      An Execution Resource in its state when RunJob returns.
    """
    messages = self.messages_module
    run_job_request = messages.RunJobRequest()
    if overrides:
        run_job_request.overrides = overrides
    run_request = messages.RunNamespacesJobsRunRequest(name=job_ref.RelativeName(), runJobRequest=run_job_request)
    with metrics.RecordDuration(metric_names.RUN_JOB):
        try:
            execution_message = self._client.namespaces_jobs.Run(run_request)
        except api_exceptions.HttpError as e:
            if e.status_code == 429:
                raise serverless_exceptions.DeploymentFailedError('Resource exhausted error. This may mean that too many executions are already running. Please wait until one completes before creating a new one.')
            raise e
    if asyn:
        return execution.Execution(execution_message, messages)
    execution_ref = self._registry.Parse(execution_message.metadata.name, params={'namespacesId': execution_message.metadata.namespace}, collection='run.namespaces.executions')
    getter = functools.partial(self.GetExecution, execution_ref)
    terminal_condition = execution.COMPLETED_CONDITION if wait else execution.STARTED_CONDITION
    ex = self.GetExecution(execution_ref)
    for msg in run_condition.GetNonTerminalMessages(ex.conditions, ignore_retry=True):
        tracker.AddWarning(msg)
    poller = op_pollers.ExecutionConditionPoller(getter, tracker, terminal_condition, dependencies=stages.ExecutionDependencies())
    try:
        self.WaitForCondition(poller, None if wait else 0)
    except serverless_exceptions.ExecutionFailedError:
        raise serverless_exceptions.ExecutionFailedError('The execution failed.' + messages_util.GetExecutionCreatedMessage(release_track, ex))
    return self.GetExecution(execution_ref)