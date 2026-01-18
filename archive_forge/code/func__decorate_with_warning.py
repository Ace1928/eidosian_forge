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
def _decorate_with_warning(func: _F, wtype: Type[exc.SADeprecationWarning], message: str, version: str, docstring_header: Optional[str]=None, enable_warnings: bool=True) -> _F:
    """Wrap a function with a warnings.warn and augmented docstring."""
    message = _sanitize_restructured_text(message)
    if issubclass(wtype, exc.Base20DeprecationWarning):
        doc_only = ' (Background on SQLAlchemy 2.0 at: :ref:`migration_20_toplevel`)'
    else:
        doc_only = ''

    @decorator
    def warned(fn: _F, *args: Any, **kwargs: Any) -> _F:
        skip_warning = not enable_warnings or kwargs.pop('_sa_skip_warning', False)
        if not skip_warning:
            _warn_with_version(message, version, wtype, stacklevel=3)
        return fn(*args, **kwargs)
    doc = func.__doc__ is not None and func.__doc__ or ''
    if docstring_header is not None:
        docstring_header %= dict(func=func.__name__)
        docstring_header += doc_only
        doc = inject_docstring_text(doc, docstring_header, 1)
    decorated = warned(func)
    decorated.__doc__ = doc
    decorated._sa_warn = lambda: _warn_with_version(message, version, wtype, stacklevel=3)
    return decorated