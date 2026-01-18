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
class VirtualField(MetaField):
    field_class = None

    def __init__(self, field_class=None, *args, **kwargs):
        Field = field_class if field_class is not None else self.field_class
        self.field_instance = Field() if Field is not None else None
        super(VirtualField, self).__init__(*args, **kwargs)

    def db_value(self, value):
        if self.field_instance is not None:
            return self.field_instance.db_value(value)
        return value

    def python_value(self, value):
        if self.field_instance is not None:
            return self.field_instance.python_value(value)
        return value

    def bind(self, model, name, set_attribute=True):
        self.model = model
        self.column_name = self.name = self.safe_name = name
        setattr(model, name, self.accessor_class(model, self, name))