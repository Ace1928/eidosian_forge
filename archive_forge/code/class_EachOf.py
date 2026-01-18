from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
class EachOf(AssertRule):

    def __init__(self, *rules):
        self.rules = list(rules)

    def process_statement(self, execute_observed):
        if not self.rules:
            self.is_consumed = True
            self.consume_statement = False
        while self.rules:
            rule = self.rules[0]
            rule.process_statement(execute_observed)
            if rule.is_consumed:
                self.rules.pop(0)
            elif rule.errormessage:
                self.errormessage = rule.errormessage
            if rule.consume_statement:
                break
        if not self.rules:
            self.is_consumed = True

    def no_more_statements(self):
        if self.rules and (not self.rules[0].is_consumed):
            self.rules[0].no_more_statements()
        elif self.rules:
            super().no_more_statements()