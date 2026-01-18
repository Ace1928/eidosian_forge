from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import shutil
import textwrap
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.command_lib.builds import submit_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
@staticmethod
def ResumeUnsupportedSDK(job_id, experiment_with_token, project_id=None, region_id=None):
    """Resumes a job by calling the Jobs.Update method.

    Args:
      job_id: Identifies a single job.
      experiment_with_token: The resume token unique to the job prefixed with
        the experiment key.
      project_id: The project which owns the job.
      region_id: The regional endpoint where the job lives.

    Returns:
      (Job)
    """
    project_id = project_id or GetProject()
    region_id = region_id or DATAFLOW_API_DEFAULT_REGION
    environment = GetMessagesModule().Environment(experiments=[experiment_with_token])
    job = GetMessagesModule().Job(environment=environment)
    request = GetMessagesModule().DataflowProjectsLocationsJobsUpdateRequest(jobId=job_id, location=region_id, projectId=project_id, job=job)
    try:
        return Jobs.GetService().Update(request)
    except apitools_exceptions.HttpError as error:
        raise exceptions.HttpException(error)