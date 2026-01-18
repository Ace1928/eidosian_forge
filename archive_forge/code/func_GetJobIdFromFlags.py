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
def GetJobIdFromFlags() -> Optional[bq_client_utils.JobIdGenerator]:
    """Returns the job id or job generator from the flags."""
    if FLAGS.fingerprint_job_id and FLAGS.job_id:
        raise app.UsageError('The fingerprint_job_id flag cannot be specified with the job_id flag.')
    if FLAGS.fingerprint_job_id:
        return bq_client_utils.JobIdGeneratorFingerprint()
    elif FLAGS.job_id is None:
        return bq_client_utils.JobIdGeneratorIncrementing(bq_client_utils.JobIdGeneratorRandom())
    elif FLAGS.job_id:
        return FLAGS.job_id
    else:
        return None