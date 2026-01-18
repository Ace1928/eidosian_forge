from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
def _add_snake_case(classname):
    """Added snake_cased method from CammelCased method."""
    snake_map = {}
    for k, v in classname.__dict__.items():
        if re.match('^[A-Z]+', k):
            snake = re.sub('(?<!^)(?=[A-Z])', '_', k).lower().replace('n_best', 'nbest')
            snake_map[snake] = v
    for k, v in snake_map.items():
        setattr(classname, k, v)