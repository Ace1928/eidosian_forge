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
@private
class ImplicitBinding(Binding):
    """A binding that was created implicitly by auto-binding."""
    pass