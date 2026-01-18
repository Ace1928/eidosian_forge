import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
def _negation_str(self, config):
    if self.description is not None:
        return 'Not ' + self._format_description(config)
    else:
        return self._eval_str(config, negate=True)