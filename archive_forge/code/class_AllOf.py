from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
class AllOf(AssertRule):

    def __init__(self, *rules):
        self.rules = set(rules)

    def process_statement(self, execute_observed):
        for rule in list(self.rules):
            rule.errormessage = None
            rule.process_statement(execute_observed)
            if rule.is_consumed:
                self.rules.discard(rule)
                if not self.rules:
                    self.is_consumed = True
                break
            elif not rule.errormessage:
                self.errormessage = None
                break
        else:
            self.errormessage = list(self.rules)[0].errormessage