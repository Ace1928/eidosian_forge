from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
class CountStatements(AssertRule):

    def __init__(self, count):
        self.count = count
        self._statement_count = 0

    def process_statement(self, execute_observed):
        self._statement_count += 1

    def no_more_statements(self):
        if self.count != self._statement_count:
            assert False, 'desired statement count %d does not match %d' % (self.count, self._statement_count)