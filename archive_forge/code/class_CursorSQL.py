from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
class CursorSQL(SQLMatchRule):

    def __init__(self, statement, params=None, consume_statement=True):
        self.statement = statement
        self.params = params
        self.consume_statement = consume_statement

    def process_statement(self, execute_observed):
        stmt = execute_observed.statements[0]
        if self.statement != stmt.statement or (self.params is not None and self.params != stmt.parameters):
            self.consume_statement = True
            self.errormessage = 'Testing for exact SQL %s parameters %s received %s %s' % (self.statement, self.params, stmt.statement, stmt.parameters)
        else:
            execute_observed.statements.pop(0)
            self.is_consumed = True
            if not execute_observed.statements:
                self.consume_statement = True