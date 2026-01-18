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
def _build_class(self, cls: Type[T], **kwargs: Any) -> T:
    return self._injector.create_object(cls, additional_kwargs=kwargs)