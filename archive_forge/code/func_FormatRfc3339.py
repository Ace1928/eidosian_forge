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
def FormatRfc3339(datetime_obj):
    """Formats a datetime.datetime object (UTC) in RFC3339.

  https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#timestamp

  Args:
    datetime_obj: A datetime.datetime object representing a datetime in UTC.

  Returns:
    The string representation of the date in RFC3339.
  """
    return datetime_obj.isoformat('T') + 'Z'