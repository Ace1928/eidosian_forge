from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
def _failure_message(self, execute_observed, expected_params):
    return 'Testing for compiled statement\n%r partial params %s, received\n%%(received_statement)r with params %%(received_parameters)r' % (self._dialect_adjusted_statement(execute_observed.context.dialect).replace('%', '%%'), repr(expected_params).replace('%', '%%'))