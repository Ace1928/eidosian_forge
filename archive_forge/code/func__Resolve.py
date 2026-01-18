from __future__ import absolute_import
from __future__ import print_function
from collections import namedtuple
import copy
import hashlib
import os
import six
def _Resolve(value, environment):
    """Resolves environment variables embedded in the given value."""
    outer_env = os.environ
    try:
        os.environ = environment
        return os.path.expandvars(value)
    finally:
        os.environ = outer_env