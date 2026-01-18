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
def ListModels(self, reference, max_results, page_token):
    """Lists models for the given dataset reference.

    Arguments:
      reference: Reference to the dataset.
      max_results: Number of results to return.
      page_token: Token to retrieve the next page of results.

    Returns:
      A dict that contains entries:
        'results': a list of models
        'token': nextPageToken for the last page, if present.
    """
    return self.GetModelsApiClient().models().list(projectId=reference.projectId, datasetId=reference.datasetId, maxResults=max_results, pageToken=page_token).execute()