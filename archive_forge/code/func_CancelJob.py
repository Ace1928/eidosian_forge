from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional
import uuid
from absl import flags
from google.api_core.iam import Policy
from googleapiclient import http as http_request
import inflection
from clients import bigquery_client
from clients import client_dataset
from clients import client_reservation
from clients import table_reader as bq_table_reader
from clients import utils as bq_client_utils
from utils import bq_api_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
def CancelJob(self, project_id=None, job_id=None, location=None):
    """Attempt to cancel the specified job if it is running.

    Args:
      project_id: The project_id to the job is running under. If None,
        self.project_id is used.
      job_id: The job id for this job.
      location: Optional. The geographic location of the job.

    Returns:
      The job resource returned for the job for which cancel is being requested.

    Raises:
      bq_error.BigqueryClientConfigurationError: if project_id or job_id
        are None.
    """
    project_id = project_id or self.project_id
    if not project_id:
        raise bq_error.BigqueryClientConfigurationError('Cannot cancel a job without a project id.')
    if not job_id:
        raise bq_error.BigqueryClientConfigurationError('Cannot cancel a job without a job id.')
    job_reference = bq_id_utils.ApiClientHelper.JobReference.Create(projectId=project_id, jobId=job_id, location=location)
    result = self.apiclient.jobs().cancel(**dict(job_reference)).execute()['job']
    if result['status']['state'] != 'DONE' and self.sync:
        job_reference = bq_processor_utils.ConstructObjectReference(result)
        result = self.WaitJob(job_reference=job_reference)
    return result