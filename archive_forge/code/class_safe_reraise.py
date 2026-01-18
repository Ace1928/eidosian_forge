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
class safe_reraise:
    """Reraise an exception after invoking some
    handler code.

    Stores the existing exception info before
    invoking so that it is maintained across a potential
    coroutine context switch.

    e.g.::

        try:
            sess.commit()
        except:
            with safe_reraise():
                sess.rollback()

    TODO: we should at some point evaluate current behaviors in this regard
    based on current greenlet, gevent/eventlet implementations in Python 3, and
    also see the degree to which our own asyncio (based on greenlet also) is
    impacted by this. .rollback() will cause IO / context switch to occur in
    all these scenarios; what happens to the exception context from an
    "except:" block if we don't explicitly store it? Original issue was #2703.

    """
    __slots__ = ('_exc_info',)
    _exc_info: Union[None, Tuple[Type[BaseException], BaseException, types.TracebackType], Tuple[None, None, None]]

    def __enter__(self) -> None:
        self._exc_info = sys.exc_info()

    def __exit__(self, type_: Optional[Type[BaseException]], value: Optional[BaseException], traceback: Optional[types.TracebackType]) -> NoReturn:
        assert self._exc_info is not None
        if type_ is None:
            exc_type, exc_value, exc_tb = self._exc_info
            assert exc_value is not None
            self._exc_info = None
            raise exc_value.with_traceback(exc_tb)
        else:
            self._exc_info = None
            assert value is not None
            raise value.with_traceback(traceback)