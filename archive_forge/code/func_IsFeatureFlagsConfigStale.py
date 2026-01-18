from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import hashlib
import logging
import os
import threading
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files as file_utils
def IsFeatureFlagsConfigStale(path):
    try:
        return time.time() - os.path.getmtime(path) > _FEATURE_FLAG_CACHE_TIME_SECONDS
    except OSError:
        return True