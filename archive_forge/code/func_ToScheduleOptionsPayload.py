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
def ToScheduleOptionsPayload(self, options_to_copy=None):
    """Returns a dictionary of schedule options.

    Args:
      options_to_copy: Existing options to be copied.

    Returns:
      A dictionary of schedule options expected by the
      bigquery.transfers.create and bigquery.transfers.update API methods.
    """
    options = dict(options_to_copy or {})
    if self.start_time is not None:
        options['startTime'] = self._TimeOrInfitity(self.start_time)
    if self.end_time is not None:
        options['endTime'] = self._TimeOrInfitity(self.end_time)
    options['disableAutoScheduling'] = self.disable_auto_scheduling
    return options