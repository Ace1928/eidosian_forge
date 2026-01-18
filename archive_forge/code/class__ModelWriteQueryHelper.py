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
class _ModelWriteQueryHelper(_ModelQueryHelper):

    def __init__(self, model, *args, **kwargs):
        self.model = model
        super(_ModelWriteQueryHelper, self).__init__(model, *args, **kwargs)

    def returning(self, *returning):
        accum = []
        for item in returning:
            if is_model(item):
                accum.extend(item._meta.sorted_fields)
            else:
                accum.append(item)
        return super(_ModelWriteQueryHelper, self).returning(*accum)

    def _set_table_alias(self, ctx):
        table = self.model._meta.table
        ctx.alias_manager[table] = table.__name__