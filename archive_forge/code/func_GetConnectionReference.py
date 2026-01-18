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
def GetConnectionReference(self, identifier=None, path=None, default_location=None, default_connection_id=None):
    """Determine a ConnectionReference from an identifier and location."""
    return bq_client_utils.GetConnectionReference(id_fallbacks=self, identifier=identifier, path=path, default_location=default_location, default_connection_id=default_connection_id)