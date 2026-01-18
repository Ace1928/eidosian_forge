from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import retry
import six
def WaitForJobTermination(dataproc, job, job_ref, message, goal_state, error_state=None, stream_driver_log=False, log_poll_period_s=1, dataproc_poll_period_s=10, timeout_s=None):
    """Poll dataproc Job until its status is terminal or timeout reached.

  Args:
    dataproc: wrapper for dataproc resources, client and messages
    job: The job to wait to finish.
    job_ref: Parsed dataproc.projects.regions.jobs resource containing a
      projectId, region, and jobId.
    message: str, message to display to user while polling.
    goal_state: JobStatus.StateValueValuesEnum, the state to define success
    error_state: JobStatus.StateValueValuesEnum, the state to define failure
    stream_driver_log: bool, Whether to show the Job's driver's output.
    log_poll_period_s: number, delay in seconds between checking on the log.
    dataproc_poll_period_s: number, delay in seconds between requests to the
      Dataproc API.
    timeout_s: number, time out for job completion. None means no timeout.

  Returns:
    Job: the return value of the last successful jobs.get request.

  Raises:
    JobError: if the job finishes with an error.
  """
    request = dataproc.messages.DataprocProjectsRegionsJobsGetRequest(projectId=job_ref.projectId, region=job_ref.region, jobId=job_ref.jobId)
    driver_log_stream = None
    last_job_poll_time = 0
    job_complete = False
    wait_display = None
    driver_output_uri = None

    def ReadDriverLogIfPresent():
        if driver_log_stream and driver_log_stream.open:
            driver_log_stream.ReadIntoWritable(log.err)

    def PrintEqualsLine():
        attr = console_attr.GetConsoleAttr()
        log.err.Print('=' * attr.GetTermSize()[0])
    if stream_driver_log:
        log.status.Print('Waiting for job output...')
        wait_display = NoOpProgressDisplay()
    else:
        wait_display = progress_tracker.ProgressTracker(message, autotick=True)
    start_time = now = time.time()
    with wait_display:
        while not timeout_s or timeout_s > now - start_time:
            ReadDriverLogIfPresent()
            log_stream_closed = driver_log_stream and (not driver_log_stream.open)
            if not job_complete and job.status.state in dataproc.terminal_job_states:
                job_complete = True
                timeout_s = now - start_time + 10
            if job_complete and (not stream_driver_log or log_stream_closed):
                break
            regular_job_poll = not job_complete and now >= last_job_poll_time + dataproc_poll_period_s
            expecting_output_stream = stream_driver_log and (not driver_log_stream)
            expecting_job_done = not job_complete and log_stream_closed
            if regular_job_poll or expecting_output_stream or expecting_job_done:
                last_job_poll_time = now
                try:
                    job = dataproc.client.projects_regions_jobs.Get(request)
                except apitools_exceptions.HttpError as error:
                    log.warning('GetJob failed:\n{}'.format(six.text_type(error)))
                    if IsClientHttpException(error):
                        raise
                if stream_driver_log and job.driverOutputResourceUri and (job.driverOutputResourceUri != driver_output_uri):
                    if driver_output_uri:
                        PrintEqualsLine()
                        log.warning("Job attempt failed. Streaming new attempt's output.")
                        PrintEqualsLine()
                    driver_output_uri = job.driverOutputResourceUri
                    driver_log_stream = storage_helpers.StorageObjectSeriesStream(job.driverOutputResourceUri)
            time.sleep(log_poll_period_s)
            now = time.time()
    state = job.status.state
    if state in dataproc.terminal_job_states:
        if stream_driver_log:
            if not driver_log_stream:
                log.warning('Expected job output not found.')
            elif driver_log_stream.open:
                log.warning('Job terminated, but output did not finish streaming.')
        if state is goal_state:
            return job
        if error_state and state is error_state:
            if job.status.details:
                raise exceptions.JobError('Job [{0}] failed with error:\n{1}'.format(job_ref.jobId, job.status.details))
            raise exceptions.JobError('Job [{0}] failed.'.format(job_ref.jobId))
        if job.status.details:
            log.info('Details:\n' + job.status.details)
        raise exceptions.JobError('Job [{0}] entered state [{1}] while waiting for [{2}].'.format(job_ref.jobId, state, goal_state))
    raise exceptions.JobTimeoutError('Job [{0}] timed out while in state [{1}].'.format(job_ref.jobId, state))