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
def PrintFields(fields, indent=0):
    """Print all fields in a schema, recurring as necessary."""
    lines = []
    for field in fields:
        prefix = '|  ' * indent
        junction = '|' if field.get('type', 'STRING') != 'RECORD' else '+'
        entry = '%s- %s: %s' % (junction, field['name'], field.get('type', 'STRING').lower())
        if 'maxLength' in field:
            entry += '(%s)' % field['maxLength']
        elif 'precision' in field:
            if 'scale' in field:
                entry += '(%s, %s)' % (field['precision'], field['scale'])
            else:
                entry += '(%s)' % field['precision']
            if 'roundingMode' in field:
                entry += ' options(rounding_mode="%s")' % field['roundingMode']
        if field.get('mode', 'NULLABLE') != 'NULLABLE':
            entry += ' (%s)' % (field['mode'].lower(),)
        lines.append(prefix + entry)
        if 'fields' in field:
            lines.extend(PrintFields(field['fields'], indent + 1))
    return lines