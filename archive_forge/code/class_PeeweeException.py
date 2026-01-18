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
class PeeweeException(Exception):

    def __init__(self, *args):
        if args and isinstance(args[0], Exception):
            self.orig, args = (args[0], args[1:])
        super(PeeweeException, self).__init__(*args)