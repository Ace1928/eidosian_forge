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
def PrintDryRunInfo(job):
    """Prints the dry run info."""
    num_bytes = job['statistics']['query']['totalBytesProcessed']
    num_bytes_accuracy = job['statistics']['query'].get('totalBytesProcessedAccuracy', 'PRECISE')
    if FLAGS.format in ['prettyjson', 'json']:
        bq_utils.PrintFormattedJsonObject(job)
    elif FLAGS.format == 'csv':
        print(num_bytes)
    elif job['statistics']['query'].get('statementType', '') == 'LOAD_DATA':
        print('Query successfully validated. Assuming the files are not modified, running this query will process %s files loading %s bytes of data.' % (job['statistics']['query']['loadQueryStatistics']['inputFiles'], job['statistics']['query']['loadQueryStatistics']['inputFileBytes']))
    elif num_bytes_accuracy == 'PRECISE':
        print('Query successfully validated. Assuming the tables are not modified, running this query will process %s bytes of data.' % (num_bytes,))
    elif num_bytes_accuracy == 'LOWER_BOUND':
        print('Query successfully validated. Assuming the tables are not modified, running this query will process lower bound of %s bytes of data.' % (num_bytes,))
    elif num_bytes_accuracy == 'UPPER_BOUND':
        print('Query successfully validated. Assuming the tables are not modified, running this query will process upper bound of %s bytes of data.' % (num_bytes,))
    elif job['statistics']['query']['statementType'] == 'CREATE_MODEL':
        print('Query successfully validated. The number of bytes that will be processed by this query cannot be calculated automatically. More information about this can be seen in https://cloud.google.com/bigquery-ml/pricing#dry_run')
    else:
        print('Query successfully validated. Assuming the tables are not modified, running this query will process %s of data and the accuracy is unknown because of federated tables or clustered tables.' % (num_bytes,))