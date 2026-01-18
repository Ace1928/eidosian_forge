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
def ExecuteJob(self, configuration, sync=None, project_id=None, upload_file=None, job_id=None, location=None):
    """Execute a job, possibly waiting for results."""
    if sync is None:
        sync = self.sync
    if sync:
        job = self.RunJobSynchronously(configuration, project_id=project_id, upload_file=upload_file, job_id=job_id, location=location)
    else:
        job = self.StartJob(configuration, project_id=project_id, upload_file=upload_file, job_id=job_id, location=location)
        bq_client_utils.RaiseIfJobError(job)
    return job