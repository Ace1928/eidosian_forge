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
class NoScope(Scope):
    """An unscoped provider."""

    def get(self, unused_key: Type[T], provider: Provider[T]) -> Provider[T]:
        return provider