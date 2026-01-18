from collections import namedtuple
import sqlparse
from django.db import DatabaseError
from django.db.backends.base.introspection import BaseDatabaseIntrospection
from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
from django.db.backends.base.introspection import TableInfo
from django.db.models import Index
from django.utils.regex_helper import _lazy_re_compile
def _parse_table_constraints(self, sql, columns):
    statement = sqlparse.parse(sql)[0]
    constraints = {}
    unnamed_constrains_index = 0
    tokens = (token for token in statement.flatten() if not token.is_whitespace)
    for token in tokens:
        if token.match(sqlparse.tokens.Punctuation, '('):
            break
    while True:
        constraint_name, unique, check, end_token = self._parse_column_or_constraint_definition(tokens, columns)
        if unique:
            if constraint_name:
                constraints[constraint_name] = unique
            else:
                unnamed_constrains_index += 1
                constraints['__unnamed_constraint_%s__' % unnamed_constrains_index] = unique
        if check:
            if constraint_name:
                constraints[constraint_name] = check
            else:
                unnamed_constrains_index += 1
                constraints['__unnamed_constraint_%s__' % unnamed_constrains_index] = check
        if end_token.match(sqlparse.tokens.Punctuation, ')'):
            break
    return constraints