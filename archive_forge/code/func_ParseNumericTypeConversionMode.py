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
def ParseNumericTypeConversionMode(numeric_type_conversion_mode: Optional[str]=None) -> Optional[str]:
    """Parses the numeric type conversion mode from the arguments.

  Args:
    numeric_type_conversion_mode: specifies how the numeric values are handled
      when the value is out of scale.

  Returns:
    The conversion mode.

  Raises:
    UsageError: when an illegal value is passed.
  """
    if numeric_type_conversion_mode is None:
        return None
    elif numeric_type_conversion_mode == 'ROUND':
        return 'NUMERIC_TYPE_VALUE_ROUND'
    else:
        raise app.UsageError('Error parsing numeric_type_conversion_mode, only ROUND or no value are accepted')