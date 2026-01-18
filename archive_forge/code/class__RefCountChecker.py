from __future__ import print_function
import os
import sys
import gc
from functools import wraps
import unittest
import objgraph
class _RefCountChecker(object):
    IGNORED_TYPES = ()

    def __init__(self, testcase, function):
        self.testcase = testcase
        self.function = function
        self.deltas = []
        self.peak_stats = {}
        self.ignored_types = ()
        self.needs_setUp = False

    def _include_object_p(self, obj):
        if obj is self:
            return False
        kind = type(obj)
        if kind == type(self._include_object_p):
            try:
                exact_method_equals = self._include_object_p.__eq__(obj)
            except AttributeError:
                exact_method_equals = self._include_object_p.__cmp__(obj) == 0
            if exact_method_equals is not NotImplemented and exact_method_equals:
                return False
        for x in self.__dict__.values():
            if obj is x:
                return False
        if kind in self.ignored_types or kind in self.IGNORED_TYPES:
            return False
        return True

    def _growth(self):
        return objgraph.growth(limit=None, peak_stats=self.peak_stats, filter=self._include_object_p)

    def _report_diff(self, growth):
        if not growth:
            return '<Unable to calculate growth>'
        lines = []
        width = max((len(name) for name, _, _ in growth))
        for name, count, delta in growth:
            lines.append('%-*s%9d %+9d' % (width, name, count, delta))
        diff = '\n'.join(lines)
        return diff

    def _run_test(self, args, kwargs):
        gc_enabled = gc.isenabled()
        gc.disable()
        if self.needs_setUp:
            self.testcase.setUp()
            self.testcase.skipTearDown = False
        try:
            self.function(self.testcase, *args, **kwargs)
        finally:
            self.testcase.tearDown()
            self.testcase.doCleanups()
            self.testcase.skipTearDown = True
            self.needs_setUp = True
            if gc_enabled:
                gc.enable()

    def _growth_after(self):
        if 'urlparse' in sys.modules:
            sys.modules['urlparse'].clear_cache()
        if 'urllib.parse' in sys.modules:
            sys.modules['urllib.parse'].clear_cache()
        return self._growth()

    def _check_deltas(self, growth):
        deltas = self.deltas
        if not deltas:
            return True
        if gc.garbage:
            raise LeakCheckError('Generated uncollectable garbage %r' % (gc.garbage,))
        if deltas[-2:] == [0, 0] and len(deltas) in (2, 3):
            return False
        if deltas[-3:] == [0, 0, 0]:
            return False
        if len(deltas) >= 4 and sum(deltas[-4:]) == 0:
            return False
        if len(deltas) >= 3 and deltas[-1] > 0 and (deltas[-1] == deltas[-2]) and (deltas[-2] == deltas[-3]):
            diff = self._report_diff(growth)
            raise LeakCheckError('refcount increased by %r\n%s' % (deltas, diff))
        if sum(deltas[-3:]) <= 0 or sum(deltas[-4:]) <= 0 or deltas[-4:].count(0) >= 2:
            limit = 11
        else:
            limit = 7
        if len(deltas) >= limit:
            raise LeakCheckError('refcount increased by %r\n%s' % (deltas, self._report_diff(growth)))
        return True

    def __call__(self, args, kwargs):
        for _ in range(3):
            gc.collect()
        expect_failure = getattr(self.function, 'fails_leakcheck', False)
        if expect_failure:
            self.testcase.expect_greenlet_leak = True
        self.ignored_types = getattr(self.function, 'leakcheck_ignore_types', ())
        growth = self._growth()
        try:
            while self._check_deltas(growth):
                self._run_test(args, kwargs)
                growth = self._growth_after()
                self.deltas.append(sum((stat[2] for stat in growth)))
        except LeakCheckError:
            if not expect_failure:
                raise
        else:
            if expect_failure:
                raise LeakCheckError('Expected %s to leak but it did not.' % (self.function,))