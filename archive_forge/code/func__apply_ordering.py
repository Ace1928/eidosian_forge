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
def _apply_ordering(self, ctx):
    if self._order_by:
        ctx.literal(' ORDER BY ').sql(CommaNodeList(self._order_by))
    if self._limit is not None or (self._offset is not None and ctx.state.limit_max):
        limit = ctx.state.limit_max if self._limit is None else self._limit
        ctx.literal(' LIMIT ').sql(limit)
    if self._offset is not None:
        ctx.literal(' OFFSET ').sql(self._offset)
    return ctx