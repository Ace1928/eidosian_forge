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
def NewField(entry):
    name, _, field_type = entry.partition(':')
    if entry.count(':') > 1 or not name.strip():
        raise bq_error.BigquerySchemaError('Invalid schema entry: %s' % (entry,))
    return {'name': name.strip(), 'type': field_type.strip().upper() or 'STRING'}