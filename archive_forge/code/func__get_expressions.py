from types import NoneType
from django.contrib.postgres.indexes import OpClass
from django.core.exceptions import ValidationError
from django.db import DEFAULT_DB_ALIAS, NotSupportedError
from django.db.backends.ddl_references import Expressions, Statement, Table
from django.db.models import BaseConstraint, Deferrable, F, Q
from django.db.models.expressions import Exists, ExpressionList
from django.db.models.indexes import IndexExpression
from django.db.models.lookups import PostgresOperatorLookup
from django.db.models.sql import Query
def _get_expressions(self, schema_editor, query):
    expressions = []
    for idx, (expression, operator) in enumerate(self.expressions):
        if isinstance(expression, str):
            expression = F(expression)
        expression = ExclusionConstraintExpression(expression, operator=operator)
        expression.set_wrapper_classes(schema_editor.connection)
        expressions.append(expression)
    return ExpressionList(*expressions).resolve_expression(query)