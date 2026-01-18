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
def ValidateGlobalFlags():
    """Validate combinations of global flag values."""
    if FLAGS.service_account and FLAGS.use_gce_service_account:
        raise app.UsageError('Cannot specify both --service_account and --use_gce_service_account.')