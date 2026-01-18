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
def PrintObjectsArray(object_infos, objects_type):
    if FLAGS.format in ['prettyjson', 'json']:
        bq_utils.PrintFormattedJsonObject(object_infos)
    elif FLAGS.format in [None, 'sparse', 'pretty']:
        if not object_infos:
            return
        formatter = GetFormatterFromFlags()
        bq_client_utils.ConfigureFormatter(formatter, objects_type, print_format='list')
        formatted_infos = list(map(functools.partial(bq_client_utils.FormatInfoByType, object_type=objects_type), object_infos))
        for info in formatted_infos:
            formatter.AddDict(info)
        formatter.Print()
    elif object_infos:
        formatter = GetFormatterFromFlags()
        formatter.AddColumns(list(object_infos[0].keys()))
        for info in object_infos:
            formatter.AddDict(info)
        formatter.Print()