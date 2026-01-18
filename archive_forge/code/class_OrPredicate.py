import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
class OrPredicate(Predicate):

    def __init__(self, predicates, description=None):
        self.predicates = predicates
        self.description = description

    def __call__(self, config):
        for pred in self.predicates:
            if pred(config):
                return True
        return False

    def _eval_str(self, config, negate=False):
        if negate:
            conjunction = ' and '
        else:
            conjunction = ' or '
        return conjunction.join((p._as_string(config, negate=negate) for p in self.predicates))

    def _negation_str(self, config):
        if self.description is not None:
            return 'Not ' + self._format_description(config)
        else:
            return self._eval_str(config, negate=True)

    def _as_string(self, config, negate=False):
        if negate:
            return self._negation_str(config)
        elif self.description is not None:
            return self._format_description(config)
        else:
            return self._eval_str(config)