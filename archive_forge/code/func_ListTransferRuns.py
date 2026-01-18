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
def ListTransferRuns(self, reference, run_attempt, max_results=None, page_token=None, states=None):
    """Return a list of transfer runs.

    Args:
      reference: The ProjectReference to list transfer runs for.
      run_attempt: Which runs should be pulled. The default value is 'LATEST',
        which only returns the latest run per day. To return all runs, please
        specify 'RUN_ATTEMPT_UNSPECIFIED'.
      max_results: The maximum number of transfer runs to return (optional).
      page_token: Current page token (optional).
      states: States to filter transfer runs (optional).

    Returns:
      A list of transfer runs.
    """
    transfer_client = self.GetTransferV1ApiClient()
    bq_id_utils.typecheck(reference, bq_id_utils.ApiClientHelper.TransferRunReference, method='ListTransferRuns')
    reference = str(reference)
    request = bq_processor_utils.PrepareTransferRunListRequest(reference, run_attempt, max_results, page_token, states)
    response = transfer_client.projects().locations().transferConfigs().runs().list(**request).execute()
    transfer_runs = response.get('transferRuns', [])
    if max_results is not None:
        while 'nextPageToken' in response and len(transfer_runs) < max_results:
            page_token = response.get('nextPageToken')
            max_results -= len(transfer_runs)
            request = bq_processor_utils.PrepareTransferRunListRequest(reference, run_attempt, max_results, page_token, states)
            response = transfer_client.projects().locations().transferConfigs().runs().list(**request).execute()
            transfer_runs.extend(response.get('transferRuns', []))
        if response.get('nextPageToken'):
            return (transfer_runs, response.get('nextPageToken'))
    return (transfer_runs,)