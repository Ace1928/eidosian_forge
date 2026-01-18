from collections import namedtuple
import sqlparse
from django.db import DatabaseError
from django.db.backends.base.introspection import BaseDatabaseIntrospection
from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
from django.db.backends.base.introspection import TableInfo
from django.db.models import Index
from django.utils.regex_helper import _lazy_re_compile
def _parse_column_or_constraint_definition(self, tokens, columns):
    token = None
    is_constraint_definition = None
    field_name = None
    constraint_name = None
    unique = False
    unique_columns = []
    check = False
    check_columns = []
    braces_deep = 0
    for token in tokens:
        if token.match(sqlparse.tokens.Punctuation, '('):
            braces_deep += 1
        elif token.match(sqlparse.tokens.Punctuation, ')'):
            braces_deep -= 1
            if braces_deep < 0:
                break
        elif braces_deep == 0 and token.match(sqlparse.tokens.Punctuation, ','):
            break
        if is_constraint_definition is None:
            is_constraint_definition = token.match(sqlparse.tokens.Keyword, 'CONSTRAINT')
            if is_constraint_definition:
                continue
        if is_constraint_definition:
            if constraint_name is None:
                if token.ttype in (sqlparse.tokens.Name, sqlparse.tokens.Keyword):
                    constraint_name = token.value
                elif token.ttype == sqlparse.tokens.Literal.String.Symbol:
                    constraint_name = token.value[1:-1]
            if token.match(sqlparse.tokens.Keyword, 'UNIQUE'):
                unique = True
                unique_braces_deep = braces_deep
            elif unique:
                if unique_braces_deep == braces_deep:
                    if unique_columns:
                        unique = False
                    continue
                if token.ttype in (sqlparse.tokens.Name, sqlparse.tokens.Keyword):
                    unique_columns.append(token.value)
                elif token.ttype == sqlparse.tokens.Literal.String.Symbol:
                    unique_columns.append(token.value[1:-1])
        else:
            if field_name is None:
                if token.ttype in (sqlparse.tokens.Name, sqlparse.tokens.Keyword):
                    field_name = token.value
                elif token.ttype == sqlparse.tokens.Literal.String.Symbol:
                    field_name = token.value[1:-1]
            if token.match(sqlparse.tokens.Keyword, 'UNIQUE'):
                unique_columns = [field_name]
        if token.match(sqlparse.tokens.Keyword, 'CHECK'):
            check = True
            check_braces_deep = braces_deep
        elif check:
            if check_braces_deep == braces_deep:
                if check_columns:
                    check = False
                continue
            if token.ttype in (sqlparse.tokens.Name, sqlparse.tokens.Keyword):
                if token.value in columns:
                    check_columns.append(token.value)
            elif token.ttype == sqlparse.tokens.Literal.String.Symbol:
                if token.value[1:-1] in columns:
                    check_columns.append(token.value[1:-1])
    unique_constraint = {'unique': True, 'columns': unique_columns, 'primary_key': False, 'foreign_key': None, 'check': False, 'index': False} if unique_columns else None
    check_constraint = {'check': True, 'columns': check_columns, 'primary_key': False, 'unique': False, 'foreign_key': None, 'index': False} if check_columns else None
    return (constraint_name, unique_constraint, check_constraint, token)