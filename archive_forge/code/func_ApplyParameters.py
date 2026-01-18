from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import re
from typing import Any, List, NamedTuple, Optional
from utils import bq_error
from utils import bq_id_utils
def ApplyParameters(config, **kwds) -> None:
    """Adds all kwds to config dict, adjusting keys to camelcase.

  Note this does not remove entries that are set to None, however.

  Args:
    config: A configuration dict.
    **kwds: A dict of keys and values to set in the config.
  """
    config.update(((ToLowerCamel(k), v) for k, v in kwds.items() if v is not None))