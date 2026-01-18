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
class NamespaceAttribute(ColumnBase):

    def __init__(self, namespace, attribute):
        self._namespace = namespace
        self._attribute = attribute

    def __sql__(self, ctx):
        return ctx.literal(self._namespace._name + '.').sql(Entity(self._attribute))