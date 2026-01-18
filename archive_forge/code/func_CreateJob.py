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
def CreateJob(self, job_ref, config_changes, tracker=None, asyn=False):
    """Create a new Cloud Run Job.

    Args:
      job_ref: Resource, the job to create.
      config_changes: list, objects that implement Adjust().
      tracker: StagedProgressTracker, to report on the progress of releasing.
      asyn: bool, if True, return without waiting for the job to be updated.

    Returns:
      A job.Job object.
    """
    messages = self.messages_module
    new_job = job.Job.New(self._client, job_ref.Parent().Name())
    new_job.name = job_ref.Name()
    parent = job_ref.Parent().RelativeName()
    for config_change in config_changes:
        new_job = config_change.Adjust(new_job)
    create_request = messages.RunNamespacesJobsCreateRequest(job=new_job.Message(), parent=parent)
    with metrics.RecordDuration(metric_names.CREATE_JOB):
        try:
            created_job = job.Job(self._client.namespaces_jobs.Create(create_request), messages)
        except api_exceptions.HttpConflictError:
            raise serverless_exceptions.DeploymentFailedError('Job [{}] already exists.'.format(job_ref.Name()))
    if not asyn:
        getter = functools.partial(self.GetJob, job_ref)
        poller = op_pollers.ConditionPoller(getter, tracker)
        self.WaitForCondition(poller)
        created_job = poller.GetResource()
    return created_job