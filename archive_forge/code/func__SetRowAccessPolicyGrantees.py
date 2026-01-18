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
def _SetRowAccessPolicyGrantees(self, row_access_policy):
    """Sets the grantees on the given Row Access Policy."""
    row_access_policy_ref = bq_id_utils.ApiClientHelper.RowAccessPolicyReference.Create(**row_access_policy['rowAccessPolicyReference'])
    iam_policy = self.GetRowAccessPolicyIAMPolicy(row_access_policy_ref)
    grantees = self._GetGranteesFromRowAccessPolicyIamPolicy(iam_policy)
    row_access_policy['grantees'] = grantees