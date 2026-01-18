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
class ClassAssistedBuilder(AssistedBuilder[T]):

    def build(self, **kwargs: Any) -> T:
        return self._build_class(self._target, **kwargs)