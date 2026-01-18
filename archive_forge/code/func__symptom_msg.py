from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def _symptom_msg(self, msg):
    """Return the structured Symptom message."""
    return 'Symptom: ' + msg