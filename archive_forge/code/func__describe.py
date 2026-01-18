import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
def _describe(c: Any) -> str:
    if hasattr(c, '__name__'):
        return cast(str, c.__name__)
    if type(c) in (tuple, list):
        return '[%s]' % c[0].__name__
    return str(c)