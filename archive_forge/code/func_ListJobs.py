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
def ListJobs(self, reference=None, max_results=None, page_token=None, state_filter=None, min_creation_time=None, max_creation_time=None, all_users=None, parent_job_id=None):
    """Return a list of jobs.

    Args:
      reference: The ProjectReference to list jobs for.
      max_results: The maximum number of jobs to return.
      page_token: Current page token (optional).
      state_filter: A single state filter or a list of filters to apply. If not
        specified, no filtering is applied.
      min_creation_time: Timestamp in milliseconds. Only return jobs created
        after or at this timestamp.
      max_creation_time: Timestamp in milliseconds. Only return jobs created
        before or at this timestamp.
      all_users: Whether to list jobs for all users of the project. Requesting
        user must be an owner of the project to list all jobs.
      parent_job_id: Retrieve only child jobs belonging to this parent; None to
        retrieve top-level jobs.

    Returns:
      A list of jobs.
    """
    return self.ListJobsAndToken(reference, max_results, page_token, state_filter, min_creation_time, max_creation_time, all_users, parent_job_id)['results']