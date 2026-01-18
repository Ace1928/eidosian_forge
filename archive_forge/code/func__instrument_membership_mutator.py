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
def _instrument_membership_mutator(method, before, argument, after):
    """Route method args and/or return value through the collection
    adapter."""
    if before:
        fn_args = list(util.flatten_iterator(inspect_getfullargspec(method)[0]))
        if isinstance(argument, int):
            pos_arg = argument
            named_arg = len(fn_args) > argument and fn_args[argument] or None
        else:
            if argument in fn_args:
                pos_arg = fn_args.index(argument)
            else:
                pos_arg = None
            named_arg = argument
        del fn_args

    def wrapper(*args, **kw):
        if before:
            if pos_arg is None:
                if named_arg not in kw:
                    raise sa_exc.ArgumentError('Missing argument %s' % argument)
                value = kw[named_arg]
            elif len(args) > pos_arg:
                value = args[pos_arg]
            elif named_arg in kw:
                value = kw[named_arg]
            else:
                raise sa_exc.ArgumentError('Missing argument %s' % argument)
        initiator = kw.pop('_sa_initiator', None)
        if initiator is False:
            executor = None
        else:
            executor = args[0]._sa_adapter
        if before and executor:
            getattr(executor, before)(value, initiator)
        if not after or not executor:
            return method(*args, **kw)
        else:
            res = method(*args, **kw)
            if res is not None:
                getattr(executor, after)(res, initiator)
            return res
    wrapper._sa_instrumented = True
    if hasattr(method, '_sa_instrument_role'):
        wrapper._sa_instrument_role = method._sa_instrument_role
    wrapper.__name__ = method.__name__
    wrapper.__doc__ = method.__doc__
    return wrapper