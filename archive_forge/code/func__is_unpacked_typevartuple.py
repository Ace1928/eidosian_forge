from abc import abstractmethod, ABCMeta
import collections
from collections import defaultdict
import collections.abc
import contextlib
import functools
import operator
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
import warnings
from types import WrapperDescriptorType, MethodWrapperType, MethodDescriptorType, GenericAlias
def _is_unpacked_typevartuple(x: Any) -> bool:
    return not isinstance(x, type) and getattr(x, '__typing_is_unpacked_typevartuple__', False)