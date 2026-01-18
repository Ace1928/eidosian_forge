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
class CombinedExpression(SQLiteNumericMixin, Expression):

    def __init__(self, lhs, connector, rhs, output_field=None):
        super().__init__(output_field=output_field)
        self.connector = connector
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return '<{}: {}>'.format(self.__class__.__name__, self)

    def __str__(self):
        return '{} {} {}'.format(self.lhs, self.connector, self.rhs)

    def get_source_expressions(self):
        return [self.lhs, self.rhs]

    def set_source_expressions(self, exprs):
        self.lhs, self.rhs = exprs

    def _resolve_output_field(self):
        combined_type = _resolve_combined_type(self.connector, type(self.lhs._output_field_or_none), type(self.rhs._output_field_or_none))
        if combined_type is None:
            raise FieldError(f'Cannot infer type of {self.connector!r} expression involving these types: {self.lhs.output_field.__class__.__name__}, {self.rhs.output_field.__class__.__name__}. You must set output_field.')
        return combined_type()

    def as_sql(self, compiler, connection):
        expressions = []
        expression_params = []
        sql, params = compiler.compile(self.lhs)
        expressions.append(sql)
        expression_params.extend(params)
        sql, params = compiler.compile(self.rhs)
        expressions.append(sql)
        expression_params.extend(params)
        expression_wrapper = '(%s)'
        sql = connection.ops.combine_expression(self.connector, expressions)
        return (expression_wrapper % sql, expression_params)

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        lhs = self.lhs.resolve_expression(query, allow_joins, reuse, summarize, for_save)
        rhs = self.rhs.resolve_expression(query, allow_joins, reuse, summarize, for_save)
        if not isinstance(self, (DurationExpression, TemporalSubtraction)):
            try:
                lhs_type = lhs.output_field.get_internal_type()
            except (AttributeError, FieldError):
                lhs_type = None
            try:
                rhs_type = rhs.output_field.get_internal_type()
            except (AttributeError, FieldError):
                rhs_type = None
            if 'DurationField' in {lhs_type, rhs_type} and lhs_type != rhs_type:
                return DurationExpression(self.lhs, self.connector, self.rhs).resolve_expression(query, allow_joins, reuse, summarize, for_save)
            datetime_fields = {'DateField', 'DateTimeField', 'TimeField'}
            if self.connector == self.SUB and lhs_type in datetime_fields and (lhs_type == rhs_type):
                return TemporalSubtraction(self.lhs, self.rhs).resolve_expression(query, allow_joins, reuse, summarize, for_save)
        c = self.copy()
        c.is_summary = summarize
        c.lhs = lhs
        c.rhs = rhs
        return c

    @cached_property
    def allowed_default(self):
        return self.lhs.allowed_default and self.rhs.allowed_default