import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
class compound:

    def __init__(self):
        self.fails = set()
        self.skips = set()

    def __add__(self, other):
        return self.add(other)

    def as_skips(self):
        rule = compound()
        rule.skips.update(self.skips)
        rule.skips.update(self.fails)
        return rule

    def add(self, *others):
        copy = compound()
        copy.fails.update(self.fails)
        copy.skips.update(self.skips)
        for other in others:
            copy.fails.update(other.fails)
            copy.skips.update(other.skips)
        return copy

    def not_(self):
        copy = compound()
        copy.fails.update((NotPredicate(fail) for fail in self.fails))
        copy.skips.update((NotPredicate(skip) for skip in self.skips))
        return copy

    @property
    def enabled(self):
        return self.enabled_for_config(config._current)

    def enabled_for_config(self, config):
        for predicate in self.skips.union(self.fails):
            if predicate(config):
                return False
        else:
            return True

    def matching_config_reasons(self, config):
        return [predicate._as_string(config) for predicate in self.skips.union(self.fails) if predicate(config)]

    def _extend(self, other):
        self.skips.update(other.skips)
        self.fails.update(other.fails)

    def __call__(self, fn):
        if hasattr(fn, '_sa_exclusion_extend'):
            fn._sa_exclusion_extend._extend(self)
            return fn

        @decorator
        def decorate(fn, *args, **kw):
            return self._do(config._current, fn, *args, **kw)
        decorated = decorate(fn)
        decorated._sa_exclusion_extend = self
        return decorated

    @contextlib.contextmanager
    def fail_if(self):
        all_fails = compound()
        all_fails.fails.update(self.skips.union(self.fails))
        try:
            yield
        except Exception as ex:
            all_fails._expect_failure(config._current, ex)
        else:
            all_fails._expect_success(config._current)

    def _do(self, cfg, fn, *args, **kw):
        for skip in self.skips:
            if skip(cfg):
                msg = "'%s' : %s" % (config.get_current_test_name(), skip._as_string(cfg))
                config.skip_test(msg)
        try:
            return_value = fn(*args, **kw)
        except Exception as ex:
            self._expect_failure(cfg, ex, name=fn.__name__)
        else:
            self._expect_success(cfg, name=fn.__name__)
            return return_value

    def _expect_failure(self, config, ex, name='block'):
        for fail in self.fails:
            if fail(config):
                print('%s failed as expected (%s): %s ' % (name, fail._as_string(config), ex))
                break
        else:
            raise ex.with_traceback(sys.exc_info()[2])

    def _expect_success(self, config, name='block'):
        if not self.fails:
            return
        for fail in self.fails:
            if fail(config):
                raise AssertionError("Unexpected success for '%s' (%s)" % (name, ' and '.join((fail._as_string(config) for fail in self.fails))))