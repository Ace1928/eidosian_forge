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
class DictCursorWrapper(CursorWrapper):

    def _initialize_columns(self):
        description = self.cursor.description
        self.columns = [t[0][t[0].rfind('.') + 1:].strip('()"`') for t in description]
        self.ncols = len(description)
    initialize = _initialize_columns

    def _row_to_dict(self, row):
        result = {}
        for i in range(self.ncols):
            result.setdefault(self.columns[i], row[i])
        return result
    process_row = _row_to_dict