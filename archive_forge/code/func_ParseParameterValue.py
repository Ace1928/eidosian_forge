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
def ParseParameterValue(type_dict, value_input):
    """Parse a parameter value of type `type_dict` from value_input.

  Arguments:
    type_dict: The JSON-dict type as which to parse `value_input`.
    value_input: Either a string representing the value, or a JSON dict for
      array and value types.
  """
    if 'structTypes' in type_dict:
        if isinstance(value_input, str):
            if value_input == 'NULL':
                return {'structValues': None}
            value_input = json.loads(value_input)
        type_map = dict([(x['name'], x['type']) for x in type_dict['structTypes']])
        values = {}
        for field_name, value in value_input.items():
            values[field_name] = ParseParameterValue(type_map[field_name], value)
        return {'structValues': values}
    if 'arrayType' in type_dict:
        if isinstance(value_input, str):
            if value_input == 'NULL':
                return {'arrayValues': None}
            try:
                value_input = json.loads(value_input)
            except json.decoder.JSONDecodeError:
                tb = sys.exc_info()[2]
                raise app.UsageError('Error parsing string as JSON: %s' % value_input).with_traceback(tb)
        values = [ParseParameterValue(type_dict['arrayType'], x) for x in value_input]
        if not values:
            return {'value': {}}
        return {'arrayValues': values}
    if 'rangeElementType' in type_dict:
        if value_input == 'NULL':
            return {'rangeValue': None}
        start, end = ParseRangeParameterValue(value_input)
        return {'rangeValue': {'start': ParseParameterValue(type_dict['rangeElementType'], start), 'end': ParseParameterValue(type_dict['rangeElementType'], end)}}
    return {'value': value_input if value_input != 'NULL' else None}