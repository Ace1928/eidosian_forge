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
class JobIdGeneratorFingerprint(JobIdGenerator):
    """Generates job ids that uniquely match the job config."""

    def _HashableRepr(self, obj):
        if isinstance(obj, bytes):
            return obj
        return str(obj).encode('utf-8')

    def _Hash(self, config, sha1):
        """Computes the sha1 hash of a dict."""
        keys = list(config.keys())
        keys.sort()
        for key in keys:
            sha1.update(self._HashableRepr(key))
            v = config[key]
            if isinstance(v, dict):
                logging.info('Hashing: %s...', key)
                self._Hash(v, sha1)
            elif isinstance(v, list):
                logging.info('Hashing: %s ...', key)
                for inner_v in v:
                    self._Hash(inner_v, sha1)
            else:
                logging.info('Hashing: %s:%s', key, v)
                sha1.update(self._HashableRepr(v))

    def Generate(self, job_configuration):
        s1 = hashlib.sha1()
        self._Hash(job_configuration, s1)
        job_id = 'bqjob_c%s' % (s1.hexdigest(),)
        logging.info('Fingerprinting: %s:\n%s', job_configuration, job_id)
        return job_id