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
class CircularDependency(Error):
    """Circular dependency detected."""