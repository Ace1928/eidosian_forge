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
@operation
def add_column_default(self, table, column, default):
    if default is None:
        raise ValueError('`default` must be not None/NULL.')
    if callable_(default):
        default = default()
    if isinstance(default, str) and (not default.endswith((')', "'"))) and (not default.isdigit()):
        default = "'%s'" % default

    def _add_default(column_name, column_def):
        return column_def + ' DEFAULT %s' % default
    return self._update_column(table, column, _add_default)