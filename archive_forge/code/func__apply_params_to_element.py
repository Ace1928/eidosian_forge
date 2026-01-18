from __future__ import annotations
import enum
from itertools import zip_longest
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from .visitors import anon_map
from .visitors import HasTraversalDispatch
from .visitors import HasTraverseInternals
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import util
from ..inspection import inspect
from ..util import HasMemoized
from ..util.typing import Literal
from ..util.typing import Protocol
@util.preload_module('sqlalchemy.sql.elements')
def _apply_params_to_element(self, original_cache_key: CacheKey, target_element: ColumnElement[Any]) -> ColumnElement[Any]:
    if target_element._is_immutable or original_cache_key is self:
        return target_element
    elements = util.preloaded.sql_elements
    return elements._OverrideBinds(target_element, self.bindparams, original_cache_key.bindparams)