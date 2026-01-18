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
def ParseParameterTypeAndValue(param_string):
    """Parse a string of the form <recursive_type>:<value> into each part."""
    type_string, value_string = SplitParam(param_string)
    if not type_string:
        type_string = 'STRING'
    type_dict = ParseParameterType(type_string)
    return (type_dict, ParseParameterValue(type_dict, value_string))