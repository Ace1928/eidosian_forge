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
def _sanitize_restructured_text(text: str) -> str:

    def repl(m: Match[str]) -> str:
        type_, name = m.group(1, 2)
        if type_ in ('func', 'meth'):
            name += '()'
        return name
    text = re.sub(':ref:`(.+) <.*>`', lambda m: '"%s"' % m.group(1), text)
    return re.sub('\\:(\\w+)\\:`~?(?:_\\w+)?\\.?(.+?)`', repl, text)