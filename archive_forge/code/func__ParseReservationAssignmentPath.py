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
def _ParseReservationAssignmentPath(path: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Parses the reservation assignment path string into its components.

  Args:
    path: String specifying the reservation assignment path in the format
      projects/<project_id>/locations/<location>/
      reservations/<reservation_id>/assignments/<assignment_id> The
      reservation_id must be that of a top level reservation.

  Returns:
    A tuple of three elements: containing project_id, location and
    reservation_assignment_id. If an element is not found, it is represented by
    None.

  Raises:
    bq_error.BigqueryError: if the path could not be parsed.
  """
    pattern = re.compile('\n  ^projects\\/(?P<project_id>[\\w:\\-.]*[\\w:\\-]+)?\n  \\/locations\\/(?P<location>[\\w\\-]+)?\n  \\/reservations\\/(?P<reservation_id>[\\w\\-]+)\n  \\/assignments\\/(?P<reservation_assignment_id>[\\w\\-_]+)$\n  ', re.X)
    match = re.search(pattern, path)
    if not match:
        raise bq_error.BigqueryError('Could not parse reservation assignment path: %s' % path)
    project_id = match.groupdict().get('project_id', None)
    location = match.groupdict().get('location', None)
    reservation_id = match.groupdict().get('reservation_id', None)
    reservation_assignment_id = match.groupdict().get('reservation_assignment_id', None)
    return (project_id, location, reservation_id, reservation_assignment_id)