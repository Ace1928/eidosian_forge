from __future__ import annotations
import collections
from collections.abc import Iterable
import textwrap
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import uuid
import warnings
from sqlalchemy.util import asbool as asbool  # noqa: F401
from sqlalchemy.util import immutabledict as immutabledict  # noqa: F401
from sqlalchemy.util import to_list as to_list  # noqa: F401
from sqlalchemy.util import unique_list as unique_list
from .compat import inspect_getfullargspec
def _fn_or_list(self, fn_or_list: Union[List[Callable[..., Any]], Callable[..., Any]]) -> Callable[..., Any]:
    if self.uselist:

        def go(*arg: Any, **kw: Any) -> None:
            if TYPE_CHECKING:
                assert isinstance(fn_or_list, Sequence)
            for fn in fn_or_list:
                fn(*arg, **kw)
        return go
    else:
        return fn_or_list