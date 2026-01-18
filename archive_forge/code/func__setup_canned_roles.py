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
def _setup_canned_roles(cls, roles, methods):
    """see if this class has "canned" roles based on a known
    collection type (dict, set, list).  Apply those roles
    as needed to the "roles" dictionary, and also
    prepare "decorator" methods

    """
    collection_type = util.duck_type_collection(cls)
    if collection_type in __interfaces:
        assert collection_type is not None
        canned_roles, decorators = __interfaces[collection_type]
        for role, name in canned_roles.items():
            roles.setdefault(role, name)
        for method, decorator in decorators.items():
            fn = getattr(cls, method, None)
            if fn and method not in methods and (not hasattr(fn, '_sa_instrumented')):
                setattr(cls, method, decorator(fn))