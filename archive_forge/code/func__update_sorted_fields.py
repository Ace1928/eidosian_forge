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
def _update_sorted_fields(self):
    self.sorted_fields = list(self._sorted_field_list)
    self.sorted_field_names = [f.name for f in self.sorted_fields]