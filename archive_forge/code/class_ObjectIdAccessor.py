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
class ObjectIdAccessor(object):
    """Gives direct access to the underlying id"""

    def __init__(self, field):
        self.field = field

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            value = instance.__data__.get(self.field.name)
            if value is None and self.field.name in instance.__rel__:
                rel_obj = instance.__rel__[self.field.name]
                value = getattr(rel_obj, self.field.rel_field.name)
            return value
        return self.field

    def __set__(self, instance, value):
        setattr(instance, self.field.name, value)