import abc
from abc import abstractmethod, abstractproperty
import collections
import contextlib
import functools
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
def _trim_name(nm):
    whitelist = ('_TypeAlias', '_ForwardRef', '_TypingBase', '_FinalTypingBase')
    if nm.startswith('_') and nm not in whitelist:
        nm = nm[1:]
    return nm