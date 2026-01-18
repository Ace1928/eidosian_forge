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
def MaybeGetSessionTempObjectName(dataset_id: str, object_id: str) -> Optional[str]:
    """If we have a session temporary object, returns the user name of the object.

  Args:
    dataset_id: Dataset of object
    object_id: Id of object

  Returns:
    If the object is a session temp object, the name of the object after
    stripping out internal stuff such as session prefix and signature encodings.

    If the object is not a session temp object, the return value is None.
  """
    if not re.fullmatch('_[0-9a-f]{40}', dataset_id):
        return None
    session_prefix_regexp = '_[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}_'
    opt_signature_encoding_regexp = '(?:_b0a98f6_.*)?'
    match = re.fullmatch(session_prefix_regexp + '(.*?)' + opt_signature_encoding_regexp, object_id)
    if not match:
        return None
    return match.group(1)