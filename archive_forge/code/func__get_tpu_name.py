from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def _get_tpu_name(tpu):
    if tpu:
        return tpu
    for e in [_GKE_ENV_VARIABLE, _DEFAULT_ENV_VARIABLE]:
        if e in os.environ:
            return os.environ[e]
    return None