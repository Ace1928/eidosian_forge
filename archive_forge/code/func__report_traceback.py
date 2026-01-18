import copy
import functools
import itertools
import sys
import types
import unittest
import warnings
from testtools.compat import reraise
from testtools import content
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.matchers._basic import _FlippedEquals
from testtools.monkey import patch
from testtools.runtest import (
from testtools.testresult import (
def _report_traceback(self, exc_info, tb_label='traceback'):
    id_gen = self._traceback_id_gens.setdefault(tb_label, itertools.count(0))
    while True:
        tb_id = next(id_gen)
        if tb_id:
            tb_label = '%s-%d' % (tb_label, tb_id)
        if tb_label not in self.getDetails():
            break
    self.addDetail(tb_label, content.TracebackContent(exc_info, self, capture_locals=getattr(self, '__testtools_tb_locals__', False)))