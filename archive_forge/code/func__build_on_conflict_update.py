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
def _build_on_conflict_update(self, on_conflict, query):
    if on_conflict._conflict_target:
        stmt = SQL('ON CONFLICT')
        target = EnclosedNodeList([Entity(col) if isinstance(col, basestring) else col for col in on_conflict._conflict_target])
        if on_conflict._conflict_where is not None:
            target = NodeList([target, SQL('WHERE'), on_conflict._conflict_where])
    else:
        stmt = SQL('ON CONFLICT ON CONSTRAINT')
        target = on_conflict._conflict_constraint
        if isinstance(target, basestring):
            target = Entity(target)
    updates = []
    if on_conflict._preserve:
        for column in on_conflict._preserve:
            excluded = NodeList((SQL('EXCLUDED'), ensure_entity(column)), glue='.')
            expression = NodeList((ensure_entity(column), SQL('='), excluded))
            updates.append(expression)
    if on_conflict._update:
        for k, v in on_conflict._update.items():
            if not isinstance(v, Node):
                if isinstance(k, basestring):
                    k = getattr(query.table, k)
                if isinstance(k, Field):
                    v = k.to_value(v)
                else:
                    v = Value(v, unpack=False)
            else:
                v = QualifiedNames(v)
            updates.append(NodeList((ensure_entity(k), SQL('='), v)))
    parts = [stmt, target, SQL('DO UPDATE SET'), CommaNodeList(updates)]
    if on_conflict._where:
        parts.extend((SQL('WHERE'), QualifiedNames(on_conflict._where)))
    return NodeList(parts)