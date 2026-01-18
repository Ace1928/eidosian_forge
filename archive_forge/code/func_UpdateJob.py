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
def UpdateJob(self, job_ref, config_changes, tracker=None, asyn=False):
    """Update an existing Cloud Run Job.

    Args:
      job_ref: Resource, the job to update.
      config_changes: list, objects that implement Adjust().
      tracker: StagedProgressTracker, to report on the progress of updating.
      asyn: bool, if True, return without waiting for the job to be updated.

    Returns:
      A job.Job object.
    """
    messages = self.messages_module
    update_job = self.GetJob(job_ref)
    if update_job is None:
        raise serverless_exceptions.JobNotFoundError('Job [{}] could not be found.'.format(job_ref.Name()))
    for config_change in config_changes:
        update_job = config_change.Adjust(update_job)
    replace_request = messages.RunNamespacesJobsReplaceJobRequest(job=update_job.Message(), name=job_ref.RelativeName())
    with metrics.RecordDuration(metric_names.UPDATE_JOB):
        returned_job = job.Job(self._client.namespaces_jobs.ReplaceJob(replace_request), messages)
    if not asyn:
        getter = functools.partial(self.GetJob, job_ref)
        poller = op_pollers.ConditionPoller(getter, tracker)
        self.WaitForCondition(poller)
        returned_job = poller.GetResource()
    return returned_job