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
class BuiltinLookup(Lookup):

    def process_lhs(self, compiler, connection, lhs=None):
        lhs_sql, params = super().process_lhs(compiler, connection, lhs)
        field_internal_type = self.lhs.output_field.get_internal_type()
        if hasattr(connection.ops.__class__, 'field_cast_sql') and connection.ops.__class__.field_cast_sql is not BaseDatabaseOperations.field_cast_sql:
            warnings.warn('The usage of DatabaseOperations.field_cast_sql() is deprecated. Implement DatabaseOperations.lookup_cast() instead.', RemovedInDjango60Warning)
            db_type = self.lhs.output_field.db_type(connection=connection)
            lhs_sql = connection.ops.field_cast_sql(db_type, field_internal_type) % lhs_sql
        lhs_sql = connection.ops.lookup_cast(self.lookup_name, field_internal_type) % lhs_sql
        return (lhs_sql, list(params))

    def as_sql(self, compiler, connection):
        lhs_sql, params = self.process_lhs(compiler, connection)
        rhs_sql, rhs_params = self.process_rhs(compiler, connection)
        params.extend(rhs_params)
        rhs_sql = self.get_rhs_op(connection, rhs_sql)
        return ('%s %s' % (lhs_sql, rhs_sql), params)

    def get_rhs_op(self, connection, rhs):
        return connection.operators[self.lookup_name] % rhs