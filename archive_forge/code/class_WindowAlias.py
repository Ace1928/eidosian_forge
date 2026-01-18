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
class WindowAlias(Node):

    def __init__(self, window):
        self.window = window

    def alias(self, window_alias):
        self.window._alias = window_alias
        return self

    def __sql__(self, ctx):
        return ctx.literal(self.window._alias or 'w')