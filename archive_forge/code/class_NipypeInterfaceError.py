import os
from contextlib import AbstractContextManager
from copy import deepcopy
from textwrap import wrap
import re
from datetime import datetime as dt
from dateutil.parser import parse as parseutc
import platform
from ... import logging, config
from ...utils.misc import is_container, rgetcwd
from ...utils.filemanip import md5, hash_infile
class NipypeInterfaceError(Exception):
    """Custom error for interfaces"""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return '{}'.format(self.value)