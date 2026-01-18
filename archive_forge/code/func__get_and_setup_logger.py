import os
import re
import time
import logging
import warnings
import subprocess
from typing import List, Type, Tuple, Union, Optional, cast
from os.path import join as pjoin
from os.path import split as psplit
from libcloud.utils.py3 import StringIO, b
from libcloud.utils.logging import ExtraLogFormatter
def _get_and_setup_logger(self):
    logger = logging.getLogger('libcloud.compute.ssh')
    path = os.getenv('LIBCLOUD_DEBUG')
    if path:
        handler = logging.FileHandler(path)
        handler.setFormatter(ExtraLogFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger