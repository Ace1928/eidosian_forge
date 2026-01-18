from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import time
from typing import List, Optional, Tuple
from absl import flags
from clients import bigquery_client
from clients import client_dataset
from clients import utils as bq_client_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from frontend import utils as frontend_utils
from utils import bq_error
from utils import bq_id_utils
Copies one table to another.

    Examples:
      bq cp dataset.old_table dataset2.new_table
      bq cp --destination_kms_key=kms_key dataset.old_table dataset2.new_table
    