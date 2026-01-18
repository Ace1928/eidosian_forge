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
def ParseRangeParameterValue(range_value: str) -> Tuple[str, str]:
    """Parse a range parameter value string into its components.

  Args:
    range_value: A range value string of the form "[<start>, <end>)".

  Returns:
    A tuple (<start>, <end>).

  Raises:
    app.UsageError: if the input range value string was not formatted correctly.
  """
    parsed = ParseRangeString(range_value)
    if parsed is None:
        raise app.UsageError(f'Invalid range parameter value: {range_value}. Expected format: "[<start>, <end>)"')
    return parsed