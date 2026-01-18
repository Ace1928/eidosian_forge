from bisect import bisect_left
from bisect import bisect_right
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isclass
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import logging
import operator
import re
import socket
import struct
import sys
import threading
import time
import uuid
import warnings
class BaseModelSelect(_ModelQueryHelper):

    def union_all(self, rhs):
        return ModelCompoundSelectQuery(self.model, self, 'UNION ALL', rhs)
    __add__ = union_all

    def union(self, rhs):
        return ModelCompoundSelectQuery(self.model, self, 'UNION', rhs)
    __or__ = union

    def intersect(self, rhs):
        return ModelCompoundSelectQuery(self.model, self, 'INTERSECT', rhs)
    __and__ = intersect

    def except_(self, rhs):
        return ModelCompoundSelectQuery(self.model, self, 'EXCEPT', rhs)
    __sub__ = except_

    def __iter__(self):
        if not self._cursor_wrapper:
            self.execute()
        return iter(self._cursor_wrapper)

    def prefetch(self, *subqueries, **kwargs):
        return prefetch(self, *subqueries, **kwargs)

    def get(self, database=None):
        clone = self.paginate(1, 1)
        clone._cursor_wrapper = None
        try:
            return clone.execute(database)[0]
        except IndexError:
            sql, params = clone.sql()
            raise self.model.DoesNotExist('%s instance matching query does not exist:\nSQL: %s\nParams: %s' % (clone.model, sql, params))

    def get_or_none(self, database=None):
        try:
            return self.get(database=database)
        except self.model.DoesNotExist:
            pass

    @Node.copy
    def group_by(self, *columns):
        grouping = []
        for column in columns:
            if is_model(column):
                grouping.extend(column._meta.sorted_fields)
            elif isinstance(column, Table):
                if not column._columns:
                    raise ValueError('Cannot pass a table to group_by() that does not have columns explicitly declared.')
                grouping.extend([getattr(column, col_name) for col_name in column._columns])
            else:
                grouping.append(column)
        self._group_by = grouping