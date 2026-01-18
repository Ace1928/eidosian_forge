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
def PrintPageToken(page_token):
    """Prints the page token in the pretty format.

  Args:
    page_token: The dictionary mapping of pageToken with string 'nextPageToken'.
  """
    formatter = GetFormatterFromFlags(secondary_format='pretty')
    bq_client_utils.ConfigureFormatter(formatter, bq_id_utils.ApiClientHelper.NextPageTokenReference)
    formatter.AddDict(page_token)
    formatter.Print()