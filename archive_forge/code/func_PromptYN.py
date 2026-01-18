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
def PromptYN(message: str) -> Optional[str]:
    """Prompts user with message, returning the key 'y', 'n', or '' on enter."""
    response = None
    while response not in ['y', 'n', '']:
        response = PromptWithDefault(message)
    return response