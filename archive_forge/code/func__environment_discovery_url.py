from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def _environment_discovery_url():
    return os.environ.get(_DISCOVERY_SERVICE_URL_ENV_VARIABLE)