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
@deconstructible(path='django.db.models.ExpressionWrapper')
class ExpressionWrapper(SQLiteNumericMixin, Expression):
    """
    An expression that can wrap another expression so that it can provide
    extra context to the inner expression, such as the output_field.
    """

    def __init__(self, expression, output_field):
        super().__init__(output_field=output_field)
        self.expression = expression

    def set_source_expressions(self, exprs):
        self.expression = exprs[0]

    def get_source_expressions(self):
        return [self.expression]

    def get_group_by_cols(self):
        if isinstance(self.expression, Expression):
            expression = self.expression.copy()
            expression.output_field = self.output_field
            return expression.get_group_by_cols()
        return super().get_group_by_cols()

    def as_sql(self, compiler, connection):
        return compiler.compile(self.expression)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.expression)

    @property
    def allowed_default(self):
        return self.expression.allowed_default