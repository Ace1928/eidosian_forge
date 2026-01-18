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
def _warnings_warn(message: Union[str, Warning], category: Optional[Type[Warning]]=None, stacklevel: int=2) -> None:
    try:
        frame = sys._getframe(stacklevel)
    except ValueError:
        stacklevel = 0
    except:
        stacklevel = 0
    else:
        stacklevel_found = warning_tag_found = False
        while frame is not None:
            if not stacklevel_found and (not re.match(_not_sa_pattern, frame.f_globals.get('__name__', ''))):
                stacklevel_found = True
            if frame.f_code in _warning_tags:
                warning_tag_found = True
                _suffix, _category = _warning_tags[frame.f_code]
                category = category or _category
                message = f'{message} ({_suffix})'
            frame = frame.f_back
            if not stacklevel_found:
                stacklevel += 1
            elif stacklevel_found and warning_tag_found:
                break
    if category is not None:
        warnings.warn(message, category, stacklevel=stacklevel + 1)
    else:
        warnings.warn(message, stacklevel=stacklevel + 1)