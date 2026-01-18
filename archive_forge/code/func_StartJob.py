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
def StartJob(self, configuration, project_id=None, upload_file=None, job_id=None, location=None):
    """Start a job with the given configuration.

    Args:
      configuration: The configuration for a job.
      project_id: The project_id to run the job under. If None, self.project_id
        is used.
      upload_file: A file to include as a media upload to this request. Only
        valid on job requests that expect a media upload file.
      job_id: A unique job_id to use for this job. If a JobIdGenerator, a job id
        will be generated from the job configuration. If None, a unique job_id
        will be created for this request.
      location: Optional. The geographic location where the job should run.

    Returns:
      The job resource returned from the insert job request. If there is an
      error, the jobReference field will still be filled out with the job
      reference used in the request.

    Raises:
      bq_error.BigqueryClientConfigurationError: if project_id and
        self.project_id are None.
    """
    project_id = project_id or self.project_id
    if not project_id:
        raise bq_error.BigqueryClientConfigurationError('Cannot start a job without a project id.')
    configuration = configuration.copy()
    if self.job_property:
        configuration['properties'] = dict((prop.partition('=')[0::2] for prop in self.job_property))
    job_request = {'configuration': configuration}
    job_id = job_id or self.job_id_generator
    if isinstance(job_id, bq_client_utils.JobIdGenerator):
        job_id = job_id.Generate(configuration)
    if job_id is not None:
        job_reference = {'jobId': job_id, 'projectId': project_id}
        job_request['jobReference'] = job_reference
        if location:
            job_reference['location'] = location
    media_upload = ''
    if upload_file:
        resumable = self.enable_resumable_uploads
        if os.stat(upload_file).st_size == 0:
            resumable = False
        media_upload = http_request.MediaFileUpload(filename=upload_file, mimetype='application/octet-stream', resumable=resumable)
    request = self.apiclient.jobs().insert(body=job_request, media_body=media_upload, projectId=project_id)
    if upload_file and resumable:
        result = bq_client_utils.ExecuteInChunksWithProgress(request)
    else:
        result = request.execute()
    return result