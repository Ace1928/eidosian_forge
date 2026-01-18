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
class ValuesList(_HashableSource, BaseTable):

    def __init__(self, values, columns=None, alias=None):
        self._values = values
        self._columns = columns
        super(ValuesList, self).__init__(alias=alias)

    def _get_hash(self):
        return hash((self.__class__, id(self._values), self._alias))

    @Node.copy
    def columns(self, *names):
        self._columns = names

    def __sql__(self, ctx):
        if self._alias:
            ctx.alias_manager[self] = self._alias
        if ctx.scope == SCOPE_SOURCE or ctx.scope == SCOPE_NORMAL:
            with ctx(parentheses=not ctx.parentheses):
                ctx = ctx.literal('VALUES ').sql(CommaNodeList([EnclosedNodeList(row) for row in self._values]))
            if ctx.scope == SCOPE_SOURCE:
                ctx.literal(' AS ').sql(Entity(ctx.alias_manager[self]))
                if self._columns:
                    entities = [Entity(c) for c in self._columns]
                    ctx.sql(EnclosedNodeList(entities))
        else:
            ctx.sql(Entity(ctx.alias_manager[self]))
        return ctx