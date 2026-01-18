import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
import pprint
import traceback
import types
from datetime import datetime
from operator import itemgetter
import itertools
from functools import wraps
from contextlib import contextmanager
@classmethod
def _trim_traceback(cls, tb):
    while tb.tb_next:
        tb = tb.tb_next
    return tb