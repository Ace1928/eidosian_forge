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
def SplitParam(param_string):
    split = param_string.split(':', 1)
    if len(split) != 2:
        raise app.UsageError('Query parameters must be of the form: "name:type:value", ":type:value", or "name::value". An empty name produces a positional parameter. An empty type produces a STRING parameter.')
    return split