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
def _punch_through_alias(type_: Any) -> type:
    if sys.version_info < (3, 10) and getattr(type_, '__qualname__', '') == 'NewType.<locals>.new_type' or (sys.version_info >= (3, 10) and type(type_).__module__ == 'typing' and (type(type_).__name__ == 'NewType')):
        return type_.__supertype__
    else:
        return type_