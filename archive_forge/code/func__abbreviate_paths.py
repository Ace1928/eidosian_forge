import asyncio
import collections
import contextvars
import datetime as dt
import inspect
import functools
import numbers
import os
import re
import sys
import traceback
import warnings
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from numbers import Real
from textwrap import dedent
from threading import get_ident
from collections import abc
def _abbreviate_paths(pathspec, named_paths):
    """
    Given a dict of (pathname,path) pairs, removes any prefix shared by all pathnames.
    Helps keep menu items short yet unambiguous.
    """
    from os.path import commonprefix, dirname, sep
    prefix = commonprefix([dirname(name) + sep for name in named_paths.keys()] + [pathspec])
    return OrderedDict([(name[len(prefix):], path) for name, path in named_paths.items()])