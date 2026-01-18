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
def _ExecutePatchTableRequest(self, reference, table, autodetect_schema: bool=False, etag: Optional[str]=None):
    """Executes request to patch table.

    Args:
      reference: the TableReference to patch.
      table: the body of request
      autodetect_schema: an optional flag to perform autodetect of file schema.
      etag: if set, checks that etag in the existing table matches.
    """
    request = self.apiclient.tables().patch(autodetect_schema=autodetect_schema, body=table, **dict(reference))
    if etag:
        request.headers['If-Match'] = etag if etag else table['etag']
    request.execute()