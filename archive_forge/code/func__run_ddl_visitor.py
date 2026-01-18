from __future__ import annotations
from operator import attrgetter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Optional
from typing import Type
from typing import Union
from . import url as _url
from .. import util
def _run_ddl_visitor(self, visitorcallable: Type[Union[SchemaGenerator, SchemaDropper]], element: SchemaItem, **kwargs: Any) -> None:
    kwargs['checkfirst'] = False
    visitorcallable(self.dialect, self, **kwargs).traverse_single(element)