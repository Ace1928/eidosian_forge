import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
class NotPredicate(Predicate):

    def __init__(self, predicate, description=None):
        self.predicate = predicate
        self.description = description

    def __call__(self, config):
        return not self.predicate(config)

    def _as_string(self, config, negate=False):
        if self.description:
            return self._format_description(config, not negate)
        else:
            return self.predicate._as_string(config, not negate)