from __future__ import annotations
import collections
import enum
from functools import update_wrapper
import inspect
import itertools
import operator
import re
import sys
import textwrap
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
from . import _collections
from . import compat
from ._has_cy import HAS_CYEXTENSION
from .typing import Literal
from .. import exc
def _update_argspec_defaults_into_env(spec, env):
    """given a FullArgSpec, convert defaults to be symbol names in an env."""
    if spec.defaults:
        new_defaults = []
        i = 0
        for arg in spec.defaults:
            if type(arg).__module__ not in ('builtins', '__builtin__'):
                name = 'x%d' % i
                env[name] = arg
                new_defaults.append(name)
                i += 1
            else:
                new_defaults.append(arg)
        elem = list(spec)
        elem[3] = tuple(new_defaults)
        return compat.FullArgSpec(*elem)
    else:
        return spec