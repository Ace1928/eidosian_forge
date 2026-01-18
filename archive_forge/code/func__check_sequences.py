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
def _check_sequences(self, field):
    if not field.sequence or not self.database.sequences:
        raise ValueError('Sequences are either not supported, or are not defined for "%s".' % field.name)