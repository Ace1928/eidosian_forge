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
@Field.register_lookup
class Exact(FieldGetDbPrepValueMixin, BuiltinLookup):
    lookup_name = 'exact'

    def get_prep_lookup(self):
        from django.db.models.sql.query import Query
        if isinstance(self.rhs, Query):
            if self.rhs.has_limit_one():
                if not self.rhs.has_select_fields:
                    self.rhs.clear_select_clause()
                    self.rhs.add_fields(['pk'])
            else:
                raise ValueError('The QuerySet value for an exact lookup must be limited to one result using slicing.')
        return super().get_prep_lookup()

    def as_sql(self, compiler, connection):
        if isinstance(self.rhs, bool) and getattr(self.lhs, 'conditional', False) and connection.ops.conditional_expression_supported_in_where_clause(self.lhs):
            lhs_sql, params = self.process_lhs(compiler, connection)
            template = '%s' if self.rhs else 'NOT %s'
            return (template % lhs_sql, params)
        return super().as_sql(compiler, connection)