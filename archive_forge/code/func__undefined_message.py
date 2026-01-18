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
@property
def _undefined_message(self) -> str:
    """Build a message about the undefined value based on how it was
        accessed.
        """
    if self._undefined_hint:
        return self._undefined_hint
    if self._undefined_obj is missing:
        return f'{self._undefined_name!r} is undefined'
    if not isinstance(self._undefined_name, str):
        return f'{object_type_repr(self._undefined_obj)} has no element {self._undefined_name!r}'
    return f'{object_type_repr(self._undefined_obj)!r} has no attribute {self._undefined_name!r}'