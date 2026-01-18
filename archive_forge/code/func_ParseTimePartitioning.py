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
def ParseTimePartitioning(partitioning_type=None, partitioning_expiration=None, partitioning_field=None, partitioning_minimum_partition_date=None, partitioning_require_partition_filter=None):
    """Parses time partitioning from the arguments.

  Args:
    partitioning_type: type for the time partitioning. Supported types are HOUR,
      DAY, MONTH, and YEAR. The default value is DAY when other arguments are
      specified, which generates one partition per day.
    partitioning_expiration: number of seconds to keep the storage for a
      partition. A negative value clears this setting.
    partitioning_field: if not set, the table is partitioned based on the
      loading time; if set, the table is partitioned based on the value of this
      field.
    partitioning_minimum_partition_date: lower boundary of partition date for
      field based partitioning table.
    partitioning_require_partition_filter: if true, queries on the table must
      have a partition filter so not all partitions are scanned.

  Returns:
    Time partitioning if any of the arguments is not None, otherwise None.

  Raises:
    UsageError: when failed to parse.
  """
    time_partitioning = {}
    key_type = 'type'
    key_expiration = 'expirationMs'
    key_field = 'field'
    key_minimum_partition_date = 'minimumPartitionDate'
    key_require_partition_filter = 'requirePartitionFilter'
    if partitioning_type is not None:
        time_partitioning[key_type] = partitioning_type
    if partitioning_expiration is not None:
        time_partitioning[key_expiration] = partitioning_expiration * 1000
    if partitioning_field is not None:
        time_partitioning[key_field] = partitioning_field
    if partitioning_minimum_partition_date is not None:
        if partitioning_field is not None:
            time_partitioning[key_minimum_partition_date] = partitioning_minimum_partition_date
        else:
            raise app.UsageError('Need to specify --time_partitioning_field for --time_partitioning_minimum_partition_date.')
    if partitioning_require_partition_filter is not None:
        if time_partitioning:
            time_partitioning[key_require_partition_filter] = partitioning_require_partition_filter
    if time_partitioning:
        if key_type not in time_partitioning:
            time_partitioning[key_type] = 'DAY'
        if key_expiration in time_partitioning and time_partitioning[key_expiration] <= 0:
            time_partitioning[key_expiration] = None
        return time_partitioning
    else:
        return None