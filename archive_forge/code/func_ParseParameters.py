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
def ParseParameters(parameters):
    """Parses query parameters from an array of name:type:value.

  Arguments:
    parameters: An iterable of string-form query parameters: name:type:value.
      Name may be omitted to indicate a positional parameter: :type:value. Type
      may be omitted to indicate a string: name::value, or ::value.

  Returns:
    A list of query parameters in the form for the BigQuery API client.
  """
    if not parameters:
        return None
    results = []
    for param_string in parameters:
        if os.path.isfile(param_string):
            with open(param_string) as f:
                results += json.load(f)
        else:
            results.append(ParseParameter(param_string))
    return results