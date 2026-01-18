from __future__ import annotations
import operator
import threading
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from .base import NO_KEY
from .. import exc as sa_exc
from .. import util
from ..sql.base import NO_ARG
from ..util.compat import inspect_getfullargspec
from ..util.typing import Protocol
def _instrument_class(cls):
    """Modify methods in a class and install instrumentation."""
    if cls.__module__ == '__builtin__':
        raise sa_exc.ArgumentError('Can not instrument a built-in type. Use a subclass, even a trivial one.')
    roles, methods = _locate_roles_and_methods(cls)
    _setup_canned_roles(cls, roles, methods)
    _assert_required_roles(cls, roles, methods)
    _set_collection_attributes(cls, roles, methods)