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
def SetDatasetIAMPolicy(self, reference, policy):
    """Sets IAM policy for the given dataset resource.

    Arguments:
      reference: the DatasetReference for the dataset resource.
      policy: The policy string in JSON format.

    Returns:
      The updated IAM policy attached to the given dataset resource.

    Raises:
      TypeError: if reference is not a DatasetReference.
    """
    return client_dataset.SetDatasetIAMPolicy(apiclient=self.GetIAMPolicyApiClient(), reference=reference, policy=policy)