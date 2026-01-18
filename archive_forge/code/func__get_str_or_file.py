import os
import sys
import uuid
import time
from warnings import warn
from .base import GraphPluginBase, logger
from ...interfaces.base import CommandLine
def _get_str_or_file(self, arg):
    if os.path.isfile(arg):
        with open(arg) as f:
            content = f.read()
    else:
        content = arg
    return content