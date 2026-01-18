import argparse
import ctypes
import faulthandler
import hashlib
import io
import itertools
import logging
import multiprocessing
import os
import pickle
import random
import sys
import textwrap
import unittest
from collections import defaultdict
from contextlib import contextmanager
from importlib import import_module
from io import StringIO
import sqlparse
import django
from django.core.management import call_command
from django.db import connections
from django.test import SimpleTestCase, TestCase
from django.test.utils import NullTimeKeeper, TimeKeeper, iter_test_cases
from django.test.utils import setup_databases as _setup_databases
from django.test.utils import setup_test_environment
from django.test.utils import teardown_databases as _teardown_databases
from django.test.utils import teardown_test_environment
from django.utils.datastructures import OrderedSet
from django.utils.version import PY312
def _get_databases(self, suite):
    databases = {}
    for test in iter_test_cases(suite):
        test_databases = getattr(test, 'databases', None)
        if test_databases == '__all__':
            test_databases = connections
        if test_databases:
            serialized_rollback = getattr(test, 'serialized_rollback', False)
            databases.update(((alias, serialized_rollback or databases.get(alias, False)) for alias in test_databases))
    return databases