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
def ParseRangePartitioning(range_partitioning_spec=None):
    """Parses range partitioning from the arguments.

  Args:
    range_partitioning_spec: specification for range partitioning in the format
      of field,start,end,interval.

  Returns:
    Range partitioning if range_partitioning_spec is not None, otherwise None.
  Raises:
    UsageError: when the spec fails to parse.
  """
    range_partitioning = {}
    key_field = 'field'
    key_range = 'range'
    key_range_start = 'start'
    key_range_end = 'end'
    key_range_interval = 'interval'
    if range_partitioning_spec is not None:
        parts = range_partitioning_spec.split(',')
        if len(parts) != 4:
            raise app.UsageError('Error parsing range_partitioning. range_partitioning should be in the format of "field,start,end,interval"')
        range_partitioning[key_field] = parts[0]
        range_spec = {}
        range_spec[key_range_start] = parts[1]
        range_spec[key_range_end] = parts[2]
        range_spec[key_range_interval] = parts[3]
        range_partitioning[key_range] = range_spec
    if range_partitioning:
        return range_partitioning
    else:
        return None