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
class ModelRaw(_ModelQueryHelper, RawQuery):

    def __init__(self, model, sql, params, **kwargs):
        self.model = model
        self._returning = ()
        super(ModelRaw, self).__init__(sql=sql, params=params, **kwargs)

    def get(self):
        try:
            return self.execute()[0]
        except IndexError:
            sql, params = self.sql()
            raise self.model.DoesNotExist('%s instance matching query does not exist:\nSQL: %s\nParams: %s' % (self.model, sql, params))