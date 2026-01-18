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
class Shuffler:
    """
    This class implements shuffling with a special consistency property.
    Consistency means that, for a given seed and key function, if two sets of
    items are shuffled, the resulting order will agree on the intersection of
    the two sets. For example, if items are removed from an original set, the
    shuffled order for the new set will be the shuffled order of the original
    set restricted to the smaller set.
    """
    hash_algorithm = 'md5'

    @classmethod
    def _hash_text(cls, text):
        h = hashlib.new(cls.hash_algorithm, usedforsecurity=False)
        h.update(text.encode('utf-8'))
        return h.hexdigest()

    def __init__(self, seed=None):
        if seed is None:
            seed = random.randint(0, 10 ** 10 - 1)
            seed_source = 'generated'
        else:
            seed_source = 'given'
        self.seed = seed
        self.seed_source = seed_source

    @property
    def seed_display(self):
        return f'{self.seed!r} ({self.seed_source})'

    def _hash_item(self, item, key):
        text = '{}{}'.format(self.seed, key(item))
        return self._hash_text(text)

    def shuffle(self, items, key):
        """
        Return a new list of the items in a shuffled order.

        The `key` is a function that accepts an item in `items` and returns
        a string unique for that item that can be viewed as a string id. The
        order of the return value is deterministic. It depends on the seed
        and key function but not on the original order.
        """
        hashes = {}
        for item in items:
            hashed = self._hash_item(item, key)
            if hashed in hashes:
                msg = 'item {!r} has same hash {!r} as item {!r}'.format(item, hashed, hashes[hashed])
                raise RuntimeError(msg)
            hashes[hashed] = item
        return [hashes[hashed] for hashed in sorted(hashes)]