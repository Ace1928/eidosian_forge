import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
class SpecPredicate(Predicate):

    def __init__(self, db, op=None, spec=None, description=None):
        self.db = db
        self.op = op
        self.spec = spec
        self.description = description
    _ops = {'<': operator.lt, '>': operator.gt, '==': operator.eq, '!=': operator.ne, '<=': operator.le, '>=': operator.ge, 'in': operator.contains, 'between': lambda val, pair: val >= pair[0] and val <= pair[1]}

    def __call__(self, config):
        if config is None:
            return False
        engine = config.db
        if '+' in self.db:
            dialect, driver = self.db.split('+')
        else:
            dialect, driver = (self.db, None)
        if dialect and engine.name != dialect:
            return False
        if driver is not None and engine.driver != driver:
            return False
        if self.op is not None:
            assert driver is None, 'DBAPI version specs not supported yet'
            version = _server_version(engine)
            oper = hasattr(self.op, '__call__') and self.op or self._ops[self.op]
            return oper(version, self.spec)
        else:
            return True

    def _as_string(self, config, negate=False):
        if self.description is not None:
            return self._format_description(config)
        elif self.op is None:
            if negate:
                return 'not %s' % self.db
            else:
                return '%s' % self.db
        elif negate:
            return 'not %s %s %s' % (self.db, self.op, self.spec)
        else:
            return '%s %s %s' % (self.db, self.op, self.spec)