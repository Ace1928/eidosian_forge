from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import datetime
import functools
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
from absl import app
from absl import flags
import yaml
import table_formatter
import bq_utils
from clients import utils as bq_client_utils
from utils import bq_error
from utils import bq_id_utils
from pyglib import stringutil
def ValidateHivePartitioningOptions(hive_partitioning_mode):
    """Validates the string provided is one the API accepts.

  Should not receive None as an input, since that will fail the comparison.
  Args:
    hive_partitioning_mode: String representing which hive partitioning mode is
      requested.  Only 'AUTO' and 'STRINGS' are supported.
  """
    if hive_partitioning_mode not in ['AUTO', 'STRINGS', 'CUSTOM']:
        raise app.UsageError('Only the following hive partitioning modes are supported: "AUTO", "STRINGS" and "CUSTOM"')