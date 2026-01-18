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
def PromptWithDefault(message: str) -> str:
    """Prompts user with message, return key pressed or '' on enter."""
    if FLAGS.headless:
        print('Running --headless, accepting default for prompt: %s' % (message,))
        return ''
    return RawInput(message).lower()