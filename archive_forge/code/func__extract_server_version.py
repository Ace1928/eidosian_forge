from bisect import bisect_left
from bisect import bisect_right
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isclass
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import logging
import operator
import re
import socket
import struct
import sys
import threading
import time
import uuid
import warnings
def _extract_server_version(self, version):
    version = version.lower()
    if 'maria' in version:
        match_obj = re.search('(1\\d\\.\\d+\\.\\d+)', version)
    else:
        match_obj = re.search('(\\d\\.\\d+\\.\\d+)', version)
    if match_obj is not None:
        return tuple((int(num) for num in match_obj.groups()[0].split('.')))
    warnings.warn('Unable to determine MySQL version: "%s"' % version)
    return (0, 0, 0)