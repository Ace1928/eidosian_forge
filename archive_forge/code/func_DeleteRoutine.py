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
def DeleteRoutine(self, reference, ignore_not_found=False):
    """Deletes RoutineReference reference.

    Args:
      reference: the RoutineReference to delete.
      ignore_not_found: Whether to ignore "not found" errors.

    Raises:
      TypeError: if reference is not a RoutineReference.
      bq_error.BigqueryNotFoundError: if reference does not exist and
        ignore_not_found is False.
    """
    bq_id_utils.typecheck(reference, bq_id_utils.ApiClientHelper.RoutineReference, method='DeleteRoutine')
    try:
        self.GetRoutinesApiClient().routines().delete(**dict(reference)).execute()
    except bq_error.BigqueryNotFoundError:
        if not ignore_not_found:
            raise