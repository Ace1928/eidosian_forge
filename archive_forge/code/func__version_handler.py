import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def _version_handler(self, args, screen_info=None):
    del args
    del screen_info
    return get_tensorflow_version_lines(include_dependency_versions=True)