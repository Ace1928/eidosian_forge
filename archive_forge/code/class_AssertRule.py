from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
class AssertRule:
    is_consumed = False
    errormessage = None
    consume_statement = True

    def process_statement(self, execute_observed):
        pass

    def no_more_statements(self):
        assert False, 'All statements are complete, but pending assertion rules remain'