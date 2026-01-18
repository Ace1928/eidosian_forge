from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional
import uuid
from absl import flags
from google.api_core.iam import Policy
from googleapiclient import http as http_request
import inflection
from clients import bigquery_client
from clients import client_dataset
from clients import client_reservation
from clients import table_reader as bq_table_reader
from clients import utils as bq_client_utils
from utils import bq_api_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
def GetUpdateMaskRecursively(prefix, json_value):
    if not isinstance(json_value, dict) or not json_value:
        return [inflection.underscore(prefix)]
    result = []
    for name in json_value:
        new_prefix = prefix + '.' + name
        new_json_value = json_value.get(name)
        result.extend(GetUpdateMaskRecursively(new_prefix, new_json_value))
    return result