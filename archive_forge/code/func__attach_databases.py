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
def _attach_databases(self, conn):
    cursor = conn.cursor()
    for name, db in self._attached.items():
        cursor.execute('ATTACH DATABASE "%s" AS "%s"' % (db, name))
    cursor.close()