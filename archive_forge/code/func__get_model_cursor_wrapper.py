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
def _get_model_cursor_wrapper(self, cursor):
    if len(self._from_list) == 1 and (not self._joins):
        return ModelObjectCursorWrapper(cursor, self.model, self._returning, self.model)
    return ModelCursorWrapper(cursor, self.model, self._returning, self._from_list, self._joins)