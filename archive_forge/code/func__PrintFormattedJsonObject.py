from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import datetime
import hashlib
import json
import logging
import os
import random
import re
import string
import sys
import time
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
from absl import flags
import googleapiclient
import httplib2
from utils import bq_api_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
def _PrintFormattedJsonObject(obj, obj_format='json'):
    """Prints obj in a JSON format according to the format argument.

  Args:
    obj: The object to print.
    obj_format: The format to use: 'json' or 'prettyjson'.
  """
    json_formats = ['json', 'prettyjson']
    if obj_format == 'json':
        print(json.dumps(obj, separators=(',', ':')))
    elif obj_format == 'prettyjson':
        print(json.dumps(obj, sort_keys=True, indent=2))
    else:
        raise ValueError("Invalid json format for printing: '%s', expected one of: %s" % (obj_format, json_formats))