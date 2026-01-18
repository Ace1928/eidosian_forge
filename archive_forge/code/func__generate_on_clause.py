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
def _generate_on_clause(self, src, dest, to_field=None, on=None):
    meta = src._meta
    is_backref = fk_fields = False
    if dest in meta.model_refs:
        fk_fields = meta.model_refs[dest]
    elif dest in meta.model_backrefs:
        fk_fields = meta.model_backrefs[dest]
        is_backref = True
    if not fk_fields:
        if on is not None:
            return (None, False)
        raise ValueError('Unable to find foreign key between %s and %s. Please specify an explicit join condition.' % (src, dest))
    elif to_field is not None:
        target = to_field.field if isinstance(to_field, FieldAlias) else to_field
        fk_fields = [f for f in fk_fields if f is target or (is_backref and f.rel_field is to_field)]
    if len(fk_fields) == 1:
        return (fk_fields[0], is_backref)
    if on is None:
        for fk in fk_fields:
            if fk.name == dest._meta.name:
                return (fk, is_backref)
        raise ValueError('More than one foreign key between %s and %s. Please specify which you are joining on.' % (src, dest))
    to_field = None
    if isinstance(on, Expression):
        lhs, rhs = (on.lhs, on.rhs)
        fk_set = set(fk_fields)
        if isinstance(lhs, Field):
            lhs_f = lhs.field if isinstance(lhs, FieldAlias) else lhs
            if lhs_f in fk_set:
                to_field = lhs_f
        elif isinstance(rhs, Field):
            rhs_f = rhs.field if isinstance(rhs, FieldAlias) else rhs
            if rhs_f in fk_set:
                to_field = rhs_f
    return (to_field, False)