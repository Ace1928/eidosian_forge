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
def _id_list(self, model_or_id_list):
    if isinstance(model_or_id_list[0], Model):
        return [getattr(obj, self._dest_attr) for obj in model_or_id_list]
    return model_or_id_list