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
def _enable_all_warnings():
    __diag__.warn_multiple_tokens_in_named_alternation = True
    __diag__.warn_ungrouped_named_tokens_in_collection = True
    __diag__.warn_name_set_on_empty_Forward = True
    __diag__.warn_on_multiple_string_args_to_oneof = True