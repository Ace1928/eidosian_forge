import copy
import datetime
import functools
import inspect
from collections import defaultdict
from decimal import Decimal
from types import NoneType
from uuid import UUID
from django.core.exceptions import EmptyResultSet, FieldError, FullResultSet
from django.db import DatabaseError, NotSupportedError, connection
from django.db.models import fields
from django.db.models.constants import LOOKUP_SEP
from django.db.models.query_utils import Q
from django.utils.deconstruct import deconstructible
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
@deconstructible(path='django.db.models.Func')
class Func(SQLiteNumericMixin, Expression):
    """An SQL function call."""
    function = None
    template = '%(function)s(%(expressions)s)'
    arg_joiner = ', '
    arity = None

    def __init__(self, *expressions, output_field=None, **extra):
        if self.arity is not None and len(expressions) != self.arity:
            raise TypeError("'%s' takes exactly %s %s (%s given)" % (self.__class__.__name__, self.arity, 'argument' if self.arity == 1 else 'arguments', len(expressions)))
        super().__init__(output_field=output_field)
        self.source_expressions = self._parse_expressions(*expressions)
        self.extra = extra

    def __repr__(self):
        args = self.arg_joiner.join((str(arg) for arg in self.source_expressions))
        extra = {**self.extra, **self._get_repr_options()}
        if extra:
            extra = ', '.join((str(key) + '=' + str(val) for key, val in sorted(extra.items())))
            return '{}({}, {})'.format(self.__class__.__name__, args, extra)
        return '{}({})'.format(self.__class__.__name__, args)

    def _get_repr_options(self):
        """Return a dict of extra __init__() options to include in the repr."""
        return {}

    def get_source_expressions(self):
        return self.source_expressions

    def set_source_expressions(self, exprs):
        self.source_expressions = exprs

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        c = self.copy()
        c.is_summary = summarize
        for pos, arg in enumerate(c.source_expressions):
            c.source_expressions[pos] = arg.resolve_expression(query, allow_joins, reuse, summarize, for_save)
        return c

    def as_sql(self, compiler, connection, function=None, template=None, arg_joiner=None, **extra_context):
        connection.ops.check_expression_support(self)
        sql_parts = []
        params = []
        for arg in self.source_expressions:
            try:
                arg_sql, arg_params = compiler.compile(arg)
            except EmptyResultSet:
                empty_result_set_value = getattr(arg, 'empty_result_set_value', NotImplemented)
                if empty_result_set_value is NotImplemented:
                    raise
                arg_sql, arg_params = compiler.compile(Value(empty_result_set_value))
            except FullResultSet:
                arg_sql, arg_params = compiler.compile(Value(True))
            sql_parts.append(arg_sql)
            params.extend(arg_params)
        data = {**self.extra, **extra_context}
        if function is not None:
            data['function'] = function
        else:
            data.setdefault('function', self.function)
        template = template or data.get('template', self.template)
        arg_joiner = arg_joiner or data.get('arg_joiner', self.arg_joiner)
        data['expressions'] = data['field'] = arg_joiner.join(sql_parts)
        return (template % data, params)

    def copy(self):
        copy = super().copy()
        copy.source_expressions = self.source_expressions[:]
        copy.extra = self.extra.copy()
        return copy

    @cached_property
    def allowed_default(self):
        return all((expression.allowed_default for expression in self.source_expressions))