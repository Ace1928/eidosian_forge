from collections import namedtuple
import functools
import hashlib
import re
from peewee import *
from peewee import CommaNodeList
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import OP
from peewee import callable_
from peewee import sort_models
from peewee import sqlite3
from peewee import _truncate_constraint_name
def _fix_index(self, sql, column_to_update, new_column):
    parts = sql.split(column_to_update)
    if len(parts) == 2:
        return sql.replace(column_to_update, new_column)
    lhs, rhs = sql.rsplit('(', 1)
    if len(rhs.split(column_to_update)) == 2:
        return '%s(%s' % (lhs, rhs.replace(column_to_update, new_column))
    parts = rhs.rsplit(')', 1)[0].split(',')
    columns = [part.strip('"`[]\' ') for part in parts]
    clean = []
    for column in columns:
        if re.match('%s(?:[\\\'"`\\]]?\\s|$)' % column_to_update, column):
            column = new_column + column[len(column_to_update):]
        clean.append(column)
    return '%s(%s)' % (lhs, ', '.join(('"%s"' % c for c in clean)))