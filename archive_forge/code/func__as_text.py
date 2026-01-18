from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def _as_text(s):
    if isinstance(s, bytes):
        return s.decode('utf-8')
    return s