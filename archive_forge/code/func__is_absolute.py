import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def _is_absolute(url):
    return osutils.pathjoin('/foo', url) == url