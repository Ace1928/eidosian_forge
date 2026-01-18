import functools
import sys
import typing as t
from collections import abc
from itertools import chain
from markupsafe import escape  # noqa: F401
from markupsafe import Markup
from markupsafe import soft_str
from .async_utils import auto_aiter
from .async_utils import auto_await  # noqa: F401
from .exceptions import TemplateNotFound  # noqa: F401
from .exceptions import TemplateRuntimeError  # noqa: F401
from .exceptions import UndefinedError
from .nodes import EvalContext
from .utils import _PassArg
from .utils import concat
from .utils import internalcode
from .utils import missing
from .utils import Namespace  # noqa: F401
from .utils import object_type_repr
from .utils import pass_eval_context
class DebugUndefined(Undefined):
    """An undefined that returns the debug info when printed.

    >>> foo = DebugUndefined(name='foo')
    >>> str(foo)
    '{{ foo }}'
    >>> not foo
    True
    >>> foo + 42
    Traceback (most recent call last):
      ...
    jinja2.exceptions.UndefinedError: 'foo' is undefined
    """
    __slots__ = ()

    def __str__(self) -> str:
        if self._undefined_hint:
            message = f'undefined value printed: {self._undefined_hint}'
        elif self._undefined_obj is missing:
            message = self._undefined_name
        else:
            message = f'no such element: {object_type_repr(self._undefined_obj)}[{self._undefined_name!r}]'
        return f'{{{{ {message} }}}}'