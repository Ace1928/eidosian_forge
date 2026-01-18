from django.core.exceptions import FieldError, FullResultSet
from django.db.models.expressions import Col
from django.db.models.sql import compiler
class SQLAggregateCompiler(compiler.SQLAggregateCompiler, SQLCompiler):
    pass