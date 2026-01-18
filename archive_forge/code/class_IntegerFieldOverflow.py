import itertools
import math
import warnings
from django.core.exceptions import EmptyResultSet, FullResultSet
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.models.expressions import Case, Expression, Func, Value, When
from django.db.models.fields import (
from django.db.models.query_utils import RegisterLookupMixin
from django.utils.datastructures import OrderedSet
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
class IntegerFieldOverflow:
    underflow_exception = EmptyResultSet
    overflow_exception = EmptyResultSet

    def process_rhs(self, compiler, connection):
        rhs = self.rhs
        if isinstance(rhs, int):
            field_internal_type = self.lhs.output_field.get_internal_type()
            min_value, max_value = connection.ops.integer_field_range(field_internal_type)
            if min_value is not None and rhs < min_value:
                raise self.underflow_exception
            if max_value is not None and rhs > max_value:
                raise self.overflow_exception
        return super().process_rhs(compiler, connection)