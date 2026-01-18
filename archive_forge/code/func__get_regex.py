import re
import sys
import types
import copy
import os
import inspect
def _get_regex(func):
    return getattr(func, 'regex', func.__doc__)