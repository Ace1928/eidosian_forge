from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
class DialectSQL(CompiledSQL):

    def _compile_dialect(self, execute_observed):
        return execute_observed.context.dialect

    def _compare_no_space(self, real_stmt, received_stmt):
        stmt = re.sub('[\\n\\t]', '', real_stmt)
        return received_stmt == stmt

    def _received_statement(self, execute_observed):
        received_stmt, received_params = super()._received_statement(execute_observed)
        for real_stmt in execute_observed.statements:
            if self._compare_no_space(real_stmt.statement, received_stmt):
                break
        else:
            raise AssertionError("Can't locate compiled statement %r in list of statements actually invoked" % received_stmt)
        return (received_stmt, execute_observed.context.compiled_parameters)

    def _dialect_adjusted_statement(self, dialect):
        paramstyle = dialect.paramstyle
        stmt = re.sub('[\\n\\t]', '', self.statement)
        stmt = stmt.replace('::', '!!')
        if paramstyle == 'pyformat':
            stmt = re.sub(':([\\w_]+)', '%(\\1)s', stmt)
        else:
            repl = None
            if paramstyle == 'qmark':
                repl = '?'
            elif paramstyle == 'format':
                repl = '%s'
            elif paramstyle.startswith('numeric'):
                counter = itertools.count(1)
                num_identifier = '$' if paramstyle == 'numeric_dollar' else ':'

                def repl(m):
                    return f'{num_identifier}{next(counter)}'
            stmt = re.sub(':([\\w_]+)', repl, stmt)
        stmt = stmt.replace('!!', '::')
        return stmt

    def _compare_sql(self, execute_observed, received_statement):
        stmt = self._dialect_adjusted_statement(execute_observed.context.dialect)
        return received_statement == stmt

    def _failure_message(self, execute_observed, expected_params):
        return 'Testing for compiled statement\n%r partial params %s, received\n%%(received_statement)r with params %%(received_parameters)r' % (self._dialect_adjusted_statement(execute_observed.context.dialect).replace('%', '%%'), repr(expected_params).replace('%', '%%'))