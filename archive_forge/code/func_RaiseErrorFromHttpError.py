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
def RaiseErrorFromHttpError(e):
    """Raises a BigQueryError given an HttpError."""
    if e.resp.get('content-type', '').startswith('application/json'):
        content = json.loads(e.content.decode('utf-8'))
        RaiseError(content)
    else:
        error_details = ''
        if flags.FLAGS.use_regional_endpoints:
            error_details = ' The specified regional endpoint may not be supported.'
        raise bq_error.BigqueryCommunicationError('Could not connect with BigQuery server.%s\nHttp response status: %s\nHttp response content:\n%s' % (error_details, e.resp.get('status', '(unexpected)'), e.content))