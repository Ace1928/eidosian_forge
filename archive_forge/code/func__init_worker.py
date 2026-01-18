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
def _init_worker(counter, initial_settings=None, serialized_contents=None, process_setup=None, process_setup_args=None, debug_mode=None, used_aliases=None):
    """
    Switch to databases dedicated to this worker.

    This helper lives at module-level because of the multiprocessing module's
    requirements.
    """
    global _worker_id
    with counter.get_lock():
        counter.value += 1
        _worker_id = counter.value
    start_method = multiprocessing.get_start_method()
    if start_method == 'spawn':
        if process_setup and callable(process_setup):
            if process_setup_args is None:
                process_setup_args = ()
            process_setup(*process_setup_args)
        django.setup()
        setup_test_environment(debug=debug_mode)
    db_aliases = used_aliases if used_aliases is not None else connections
    for alias in db_aliases:
        connection = connections[alias]
        if start_method == 'spawn':
            connection.settings_dict.update(initial_settings[alias])
            if (value := serialized_contents.get(alias)):
                connection._test_serialized_contents = value
        connection.creation.setup_worker_connection(_worker_id)