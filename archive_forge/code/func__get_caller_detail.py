import argparse
import collections
from collections import abc
import copy
import enum
import errno
import functools
import glob
import inspect
import itertools
import logging
import os
import string
import sys
from oslo_config import iniparser
from oslo_config import sources
import oslo_config.sources._environment as _environment
from oslo_config import types
import stevedore
def _get_caller_detail(n=2):
    """Return a string describing where this is being called from.

    :param n: Number of steps up the stack to look. Defaults to ``2``.
    :type n: int
    :returns: str
    """
    if not _show_caller_details:
        return None
    s = inspect.stack()[:n + 1]
    try:
        frame = s[n]
        try:
            return frame[1]
        finally:
            del frame
    finally:
        del s