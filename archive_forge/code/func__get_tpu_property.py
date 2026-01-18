from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def _get_tpu_property(self, key):
    if self._use_api:
        metadata = self._fetch_cloud_tpu_metadata()
        return metadata.get(key)
    return None