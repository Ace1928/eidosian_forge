import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
class _TestContext(object):

    def __init__(self, loadercls):
        self.loadercls = loadercls
        self._fd = None
        self._fn = None
        self._ok = 0
        self._skip = 0
        self._fail = 0
        self._stats = defaultdict(dict)

    @property
    def stats(self):
        return self._stats

    @property
    def results(self):
        return (self._ok, self._skip, self._fail, self._stats)

    def start(self, fn, fd):
        assert not self._fn, 'unexpected ctx.start(), already started'
        assert isinstance(fd, dict)
        self._fn = fn
        self._fd = fd

    def end(self, fn=None):
        assert not fn or self._fn == fn, 'unexpected ctx.end(), fn mismatch'
        self._fn = None
        self._fd = None

    def ok(self, info):
        assert self._fn, 'unexpected ctx.ok(), fn=None'
        self._ok += 1
        self.dbg('PASS', info)
        self._incstat('ok')
        self.end(self._fn)

    def skip(self, info):
        assert self._fn, 'unexpected ctx.skip(), fn=None'
        self._skip += 1
        self.dbg('SKIP', info)
        self._incstat('skip')
        self.end(self._fn)

    def fail(self, info):
        assert self._fn, 'unexpected ctx.fail(), fn=None'
        self._fail += 1
        self.dbg('FAIL', info)
        self._incstat('fail')
        self.end(self._fn)

    def dbg(self, msgtype, info):
        assert self._fn, 'unexpected ctx.dbg(), fn=None'
        if DEBUG:
            print('{} {} {}: {}'.format(self.loadercls.__name__, msgtype, self._fn, info))

    def _incstat(self, s):
        assert self._fd, 'unexpected ctx._incstat(), fd=None'
        fd = self._fd

        def IS(key):
            self._stats.setdefault(s, defaultdict(int))[key] += 1
        IS('total')
        IS('extension:{}'.format(fd['ext']))
        IS('encoder:{}'.format(fd['encoder']))
        IS('fmtinfo:{}'.format(fd['fmtinfo']))
        IS('testname:{}'.format(fd['testname']))
        IS('testname+ext:{}+{}'.format(fd['testname'], fd['ext']))
        IS('encoder+ext:{}+{}'.format(fd['encoder'], fd['ext']))
        IS('encoder+testname:{}+{}'.format(fd['encoder'], fd['testname']))
        IS('fmtinfo+ext:{}+{}'.format(fd['fmtinfo'], fd['ext']))