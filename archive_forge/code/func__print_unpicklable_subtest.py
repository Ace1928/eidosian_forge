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
def _print_unpicklable_subtest(self, test, subtest, pickle_exc):
    print('\nSubtest failed:\n\n    test: {}\n subtest: {}\n\nUnfortunately, the subtest that failed cannot be pickled, so the parallel\ntest runner cannot handle it cleanly. Here is the pickling error:\n\n> {}\n\nYou should re-run this test with --parallel=1 to reproduce the failure\nwith a cleaner failure message.\n'.format(test, subtest, pickle_exc))