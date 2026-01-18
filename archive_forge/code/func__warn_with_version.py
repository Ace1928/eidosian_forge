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
def _warn_with_version(msg: str, version: str, type_: Type[exc.SADeprecationWarning], stacklevel: int, code: Optional[str]=None) -> None:
    warn = type_(msg, code=code)
    warn.deprecated_since = version
    _warnings_warn(warn, stacklevel=stacklevel + 1)