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
@classmethod
def FormatCapacityCommitmentInfo(cls, capacity_commitment):
    """Prepare a capacity commitment for printing.

    Arguments:
      capacity_commitment: capacity commitment to format.

    Returns:
      A dictionary of capacity commitment properties.
    """
    return bq_client_utils.FormatCapacityCommitmentInfo(capacity_commitment)