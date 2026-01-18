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
def ParseLabels(labels: List[str]) -> Dict[str, str]:
    """Parses a list of user-supplied strings representing labels.

  Args:
    labels: A list of user-supplied strings representing labels.  It is expected
      to be in the format "key:value".

  Returns:
    A dict mapping label keys to label values.

  Raises:
    UsageError: Incorrect label arguments were supplied.
  """
    labels_dict = {}
    for key_value in labels:
        k, _, v = key_value.partition(':')
        k = k.strip()
        if k in labels_dict:
            raise app.UsageError('Cannot specify label key "%s" multiple times' % k)
        if k.strip():
            labels_dict[k.strip()] = v.strip()
    return labels_dict