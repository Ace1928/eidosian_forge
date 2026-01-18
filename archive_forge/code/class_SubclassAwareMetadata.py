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
class SubclassAwareMetadata(Metadata):
    models = []

    def __init__(self, model, *args, **kwargs):
        super(SubclassAwareMetadata, self).__init__(model, *args, **kwargs)
        self.models.append(model)

    def map_models(self, fn):
        for model in self.models:
            fn(model)