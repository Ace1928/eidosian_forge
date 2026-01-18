import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
def _is_excluded(db, op, spec):
    return SpecPredicate(db, op, spec)(config._current)