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
@staticmethod
def FormatRow(fields, row, formatter):
    """Convert fields in a single row to bq-specific formatting."""
    values = [TablePrinter.NormalizeField(field, value) for field, value in zip(fields, row)]
    if not isinstance(formatter, table_formatter.JsonFormatter):
        values = map(TablePrinter.MaybeConvertToJson, values)
    if isinstance(formatter, table_formatter.CsvFormatter):
        values = ['' if value is None else value for value in values]
    elif not isinstance(formatter, table_formatter.JsonFormatter):
        values = ['NULL' if value is None else value for value in values]
    return values