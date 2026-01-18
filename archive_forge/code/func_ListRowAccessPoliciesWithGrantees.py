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
def ListRowAccessPoliciesWithGrantees(self, table_reference: 'bq_id_utils.ApiClientHelper.TableReference', page_size: int, page_token: str, max_concurrent_iam_calls: int=1) -> Dict[str, List[Any]]:
    """Lists row access policies for the given table reference.

    Arguments:
      table_reference: Reference to the table.
      page_size: Number of results to return.
      page_token: Token to retrieve the next page of results.
      max_concurrent_iam_calls: Number of concurrent calls to getIAMPolicy.

    Returns:
      A dict that contains entries:
        'rowAccessPolicies': a list of row access policies, with an additional
          'grantees' field that contains the row access policy grantees.
        'nextPageToken': nextPageToken for the next page, if present.
    """
    response = self._ListRowAccessPolicies(table_reference, page_size, page_token)
    if 'rowAccessPolicies' in response:
        row_access_policies = response['rowAccessPolicies']
        parallel.RunInParallel(function=self._SetRowAccessPolicyGrantees, list_of_kwargs_to_function=[{'row_access_policy': row_access_policy} for row_access_policy in row_access_policies], num_workers=max_concurrent_iam_calls, cancel_futures=True)
    return response