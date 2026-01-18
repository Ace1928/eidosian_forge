from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
class CompiledSQL(SQLMatchRule):

    def __init__(self, statement, params=None, dialect='default', enable_returning=True):
        self.statement = statement
        self.params = params
        self.dialect = dialect
        self.enable_returning = enable_returning

    def _compare_sql(self, execute_observed, received_statement):
        stmt = re.sub('[\\n\\t]', '', self.statement)
        return received_statement == stmt

    def _compile_dialect(self, execute_observed):
        if self.dialect == 'default':
            dialect = DefaultDialect()
            dialect.supports_default_metavalue = True
            if self.enable_returning:
                dialect.insert_returning = dialect.update_returning = dialect.delete_returning = True
                dialect.use_insertmanyvalues = True
                dialect.supports_multivalues_insert = True
                dialect.update_returning_multifrom = True
                dialect.delete_returning_multifrom = True
                assert dialect.insert_executemany_returning
            return dialect
        else:
            return url.URL.create(self.dialect).get_dialect()()

    def _received_statement(self, execute_observed):
        """reconstruct the statement and params in terms
        of a target dialect, which for CompiledSQL is just DefaultDialect."""
        context = execute_observed.context
        compare_dialect = self._compile_dialect(execute_observed)
        cache_key = None
        extracted_parameters = None
        if 'schema_translate_map' in context.execution_options:
            map_ = context.execution_options['schema_translate_map']
        else:
            map_ = None
        if isinstance(execute_observed.clauseelement, BaseDDLElement):
            compiled = execute_observed.clauseelement.compile(dialect=compare_dialect, schema_translate_map=map_)
        else:
            compiled = execute_observed.clauseelement.compile(cache_key=cache_key, dialect=compare_dialect, column_keys=context.compiled.column_keys, for_executemany=context.compiled.for_executemany, schema_translate_map=map_)
        _received_statement = re.sub('[\\n\\t]', '', str(compiled))
        parameters = execute_observed.parameters
        if not parameters:
            _received_parameters = [compiled.construct_params(extracted_parameters=extracted_parameters)]
        else:
            _received_parameters = [compiled.construct_params(m, extracted_parameters=extracted_parameters) for m in parameters]
        return (_received_statement, _received_parameters)

    def process_statement(self, execute_observed):
        context = execute_observed.context
        _received_statement, _received_parameters = self._received_statement(execute_observed)
        params = self._all_params(context)
        equivalent = self._compare_sql(execute_observed, _received_statement)
        if equivalent:
            if params is not None:
                all_params = list(params)
                all_received = list(_received_parameters)
                while all_params and all_received:
                    param = dict(all_params.pop(0))
                    for idx, received in enumerate(list(all_received)):
                        for param_key in param:
                            if param_key not in received or received[param_key] != param[param_key]:
                                break
                        else:
                            del all_received[idx]
                            break
                    else:
                        equivalent = False
                        break
                if all_params or all_received:
                    equivalent = False
        if equivalent:
            self.is_consumed = True
            self.errormessage = None
        else:
            self.errormessage = self._failure_message(execute_observed, params) % {'received_statement': _received_statement, 'received_parameters': _received_parameters}

    def _all_params(self, context):
        if self.params:
            if callable(self.params):
                params = self.params(context)
            else:
                params = self.params
            if not isinstance(params, list):
                params = [params]
            return params
        else:
            return None

    def _failure_message(self, execute_observed, expected_params):
        return 'Testing for compiled statement\n%r partial params %s, received\n%%(received_statement)r with params %%(received_parameters)r' % (self.statement.replace('%', '%%'), repr(expected_params).replace('%', '%%'))