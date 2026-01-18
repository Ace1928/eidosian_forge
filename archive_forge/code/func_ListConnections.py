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
def ListConnections(self, project_id: str, location: str, max_results: int, page_token: Optional[str]):
    """List connections in the project and location for the given reference.

    Arguments:
      project_id: Project ID.
      location: Location.
      max_results: Number of results to show.
      page_token: Token to retrieve the next page of results.

    Returns:
      List of connection objects
    """
    parent = 'projects/%s/locations/%s' % (project_id, location)
    client = self.GetConnectionV1ApiClient()
    return client.projects().locations().connections().list(parent=parent, pageToken=page_token, pageSize=max_results).execute()