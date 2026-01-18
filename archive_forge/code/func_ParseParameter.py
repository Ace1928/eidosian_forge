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
def ParseParameter(param_string):
    """Parse a string of the form <name><type>:<value> into each part."""
    name, param_string = SplitParam(param_string)
    try:
        type_dict, value_dict = ParseParameterTypeAndValue(param_string)
    except app.UsageError as e:
        print('Error parsing parameter %s: %s' % (name, e))
        sys.exit(1)
    result = {'parameterType': type_dict, 'parameterValue': value_dict}
    if name:
        result['name'] = name
    return result