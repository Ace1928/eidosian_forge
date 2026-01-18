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
class ModelDictCursorWrapper(BaseModelCursorWrapper):

    def process_row(self, row):
        result = {}
        columns, converters = (self.columns, self.converters)
        fields = self.fields
        for i in range(self.ncols):
            attr = columns[i]
            if attr in result:
                continue
            if converters[i] is not None:
                result[attr] = converters[i](row[i])
            else:
                result[attr] = row[i]
        return result