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
class ForUpdate(Node):

    def __init__(self, expr, of=None, nowait=None):
        expr = 'FOR UPDATE' if expr is True else expr
        if expr.lower().endswith('nowait'):
            expr = expr[:-7]
            nowait = True
        self._expr = expr
        if of is not None and (not isinstance(of, (list, set, tuple))):
            of = (of,)
        self._of = of
        self._nowait = nowait

    def __sql__(self, ctx):
        ctx.literal(self._expr)
        if self._of is not None:
            ctx.literal(' OF ').sql(CommaNodeList(self._of))
        if self._nowait:
            ctx.literal(' NOWAIT')
        return ctx