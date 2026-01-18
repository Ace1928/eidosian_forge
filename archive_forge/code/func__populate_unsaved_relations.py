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
def _populate_unsaved_relations(self, field_dict):
    for foreign_key_field in self._meta.refs:
        foreign_key = foreign_key_field.name
        conditions = foreign_key in field_dict and field_dict[foreign_key] is None and (self.__rel__.get(foreign_key) is not None)
        if conditions:
            setattr(self, foreign_key, getattr(self, foreign_key))
            field_dict[foreign_key] = self.__data__[foreign_key]