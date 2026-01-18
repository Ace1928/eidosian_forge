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
def ListTransferLogs(self, reference, message_type=None, max_results=None, page_token=None):
    """Return a list of transfer run logs.

    Args:
      reference: The ProjectReference to list transfer run logs for.
      message_type: Message types to return.
      max_results: The maximum number of transfer run logs to return.
      page_token: Current page token (optional).

    Returns:
      A list of transfer run logs.
    """
    transfer_client = self.GetTransferV1ApiClient()
    reference = str(reference)
    request = bq_processor_utils.PrepareListTransferLogRequest(reference, max_results=max_results, page_token=page_token, message_type=message_type)
    response = transfer_client.projects().locations().transferConfigs().runs().transferLogs().list(**request).execute()
    transfer_logs = response.get('transferMessages', [])
    if max_results is not None:
        while 'nextPageToken' in response and len(transfer_logs) < max_results:
            page_token = response['nextPageToken']
            max_results -= len(transfer_logs)
            request = bq_processor_utils.PrepareListTransferLogRequest(reference, max_results=max_results, page_token=page_token, message_type=message_type)
            response = transfer_client.projects().locations().transferConfigs().runs().transferLogs().list(**request).execute()
            transfer_logs.extend(response.get('transferMessages', []))
    if response.get('nextPageToken'):
        return (transfer_logs, response.get('nextPageToken'))
    return (transfer_logs,)