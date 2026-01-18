from __future__ import annotations
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import Match
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from . import compat
from .langhelpers import _hash_limit_string
from .langhelpers import _warnings_warn
from .langhelpers import decorator
from .langhelpers import inject_docstring_text
from .langhelpers import inject_param_text
from .. import exc
def _decorate_cls_with_warning(cls: Type[_T], constructor: Optional[str], wtype: Type[exc.SADeprecationWarning], message: str, version: str, docstring_header: Optional[str]=None) -> Type[_T]:
    doc = cls.__doc__ is not None and cls.__doc__ or ''
    if docstring_header is not None:
        if constructor is not None:
            docstring_header %= dict(func=constructor)
        if issubclass(wtype, exc.Base20DeprecationWarning):
            docstring_header += ' (Background on SQLAlchemy 2.0 at: :ref:`migration_20_toplevel`)'
        doc = inject_docstring_text(doc, docstring_header, 1)
        constructor_fn = None
        if type(cls) is type:
            clsdict = dict(cls.__dict__)
            clsdict['__doc__'] = doc
            clsdict.pop('__dict__', None)
            clsdict.pop('__weakref__', None)
            cls = type(cls.__name__, cls.__bases__, clsdict)
            if constructor is not None:
                constructor_fn = clsdict[constructor]
        else:
            cls.__doc__ = doc
            if constructor is not None:
                constructor_fn = getattr(cls, constructor)
        if constructor is not None:
            assert constructor_fn is not None
            assert wtype is not None
            setattr(cls, constructor, _decorate_with_warning(constructor_fn, wtype, message, version, None))
    return cls