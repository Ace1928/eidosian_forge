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
def _locate_roles_and_methods(cls):
    """search for _sa_instrument_role-decorated methods in
    method resolution order, assign to roles.

    """
    roles: Dict[str, str] = {}
    methods: Dict[str, Tuple[Optional[str], Optional[int], Optional[str]]] = {}
    for supercls in cls.__mro__:
        for name, method in vars(supercls).items():
            if not callable(method):
                continue
            if hasattr(method, '_sa_instrument_role'):
                role = method._sa_instrument_role
                assert role in ('appender', 'remover', 'iterator', 'converter')
                roles.setdefault(role, name)
            before: Optional[Tuple[str, int]] = None
            after: Optional[str] = None
            if hasattr(method, '_sa_instrument_before'):
                op, argument = method._sa_instrument_before
                assert op in ('fire_append_event', 'fire_remove_event')
                before = (op, argument)
            if hasattr(method, '_sa_instrument_after'):
                op = method._sa_instrument_after
                assert op in ('fire_append_event', 'fire_remove_event')
                after = op
            if before:
                methods[name] = before + (after,)
            elif after:
                methods[name] = (None, None, after)
    return (roles, methods)