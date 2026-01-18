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
def ExecuteInChunksWithProgress(request):
    """Run an apiclient request with a resumable upload, showing progress.

  Args:
    request: an apiclient request having a media_body that is a
      MediaFileUpload(resumable=True).

  Returns:
    The result of executing the request, if it succeeds.

  Raises:
    BigQueryError: on a non-retriable error or too many retriable errors.
  """
    result = None
    retriable_errors = 0
    output_token = None
    status = None
    while result is None:
        try:
            status, result = request.next_chunk()
        except googleapiclient.errors.HttpError as e:
            logging.error('HTTP Error %d during resumable media upload', e.resp.status)
            for key, value in e.resp.items():
                logging.info('  %s: %s', key, value)
            if e.resp.status in [502, 503, 504]:
                sleep_sec = 2 ** retriable_errors
                retriable_errors += 1
                if retriable_errors > 3:
                    raise
                print('Error %d, retry #%d' % (e.resp.status, retriable_errors))
                time.sleep(sleep_sec)
            else:
                RaiseErrorFromHttpError(e)
        except (httplib2.HttpLib2Error, IOError) as e:
            RaiseErrorFromNonHttpError(e)
        if status:
            output_token = _OverwriteCurrentLine('Uploaded %d%%... ' % int(status.progress() * 100), output_token)
    _OverwriteCurrentLine('Upload complete.', output_token)
    sys.stderr.write('\n')
    return result