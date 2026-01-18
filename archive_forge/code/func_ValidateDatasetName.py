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
def ValidateDatasetName(dataset_name: str) -> None:
    """A regex to ensure the dataset name is valid.


  Arguments:
    dataset_name: string name of the dataset to be validated.

  Raises:
    UsageError: An error occurred due to invalid dataset string.
  """
    is_valid = re.fullmatch('[a-zA-Z0-9\\_]{1,1024}', dataset_name)
    if not is_valid:
        raise app.UsageError('Dataset name: %s is invalid, must be letters (uppercase or lowercase), numbers, and underscores up to 1024 characters.' % dataset_name)