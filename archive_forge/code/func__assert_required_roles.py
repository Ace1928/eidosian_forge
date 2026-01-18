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
def _assert_required_roles(cls, roles, methods):
    """ensure all roles are present, and apply implicit instrumentation if
    needed

    """
    if 'appender' not in roles or not hasattr(cls, roles['appender']):
        raise sa_exc.ArgumentError('Type %s must elect an appender method to be a collection class' % cls.__name__)
    elif roles['appender'] not in methods and (not hasattr(getattr(cls, roles['appender']), '_sa_instrumented')):
        methods[roles['appender']] = ('fire_append_event', 1, None)
    if 'remover' not in roles or not hasattr(cls, roles['remover']):
        raise sa_exc.ArgumentError('Type %s must elect a remover method to be a collection class' % cls.__name__)
    elif roles['remover'] not in methods and (not hasattr(getattr(cls, roles['remover']), '_sa_instrumented')):
        methods[roles['remover']] = ('fire_remove_event', 1, None)
    if 'iterator' not in roles or not hasattr(cls, roles['iterator']):
        raise sa_exc.ArgumentError('Type %s must elect an iterator method to be a collection class' % cls.__name__)