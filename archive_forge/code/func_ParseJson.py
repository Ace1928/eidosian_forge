from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import re
from typing import Any, List, NamedTuple, Optional
from utils import bq_error
from utils import bq_id_utils
def ParseJson(json_string: str) -> Any:
    """Wrapper for standard json parsing, may throw BigQueryClientError."""
    try:
        return json.loads(json_string)
    except ValueError as e:
        raise bq_error.BigqueryClientError('Error decoding JSON from string %s: %s' % (json_string, e))