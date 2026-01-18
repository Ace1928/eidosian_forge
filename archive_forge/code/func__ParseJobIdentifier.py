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
def _ParseJobIdentifier(identifier: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parses a job identifier string into its components.

  Args:
    identifier: String specifying the job identifier in the format
      "project_id:job_id", "project_id:location.job_id", or "job_id".

  Returns:
    A tuple of three elements: containing project_id, location,
    job_id. If an element is not found, it is represented by
    None. If no elements are found, the tuple contains three None
    values.
  """
    project_id_pattern = '[\\w:\\-.]*[\\w:\\-]+'
    location_pattern = '[a-zA-Z\\-0-9]+'
    job_id_pattern = '[\\w\\-]+'
    pattern = re.compile('\n    ^((?P<project_id>%(PROJECT_ID)s)\n    :)?\n    ((?P<location>%(LOCATION)s)\n    \\.)?\n    (?P<job_id>%(JOB_ID)s)\n    $\n  ' % {'PROJECT_ID': project_id_pattern, 'LOCATION': location_pattern, 'JOB_ID': job_id_pattern}, re.X)
    match = re.search(pattern, identifier)
    if match:
        return (match.groupdict().get('project_id', None), match.groupdict().get('location', None), match.groupdict().get('job_id', None))
    return (None, None, None)