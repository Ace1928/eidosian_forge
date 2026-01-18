import collections
import os
import re
import zipfile
from absl import app
import numpy as np
from tensorflow.python.debug.lib import profiling
def _convert_watch_key_to_tensor_name(watch_key):
    return watch_key[:watch_key.rfind(':')]